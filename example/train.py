import argparse
import os
import sys
import time
import json
import torch
import torch.nn.functional as F
import utils
import tabulate
import models
from qtorch import *
from data import get_data
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler
from auto_low import resnet_lower, lower
from optim import SGDLP

num_types = ["weight", "activate", "grad", "error", "momentum"]

parser = argparse.ArgumentParser(description='SGD/SWA training')
parser.add_argument('--dataset', type=str, default='CIFAR10',
                    help='dataset name: CIFAR10 or IMAGENET12')
parser.add_argument('--name', type=str, default='', metavar='S', required=True,
                    help="Name for the log dir and checkpoint dir")
parser.add_argument('--data_path', type=str, default="./data", required=True, metavar='PATH',
                    help='path to datasets location (default: "./data")')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size (default: 128)')
parser.add_argument('--val_ratio', type=float, default=0.0, metavar='N',
                    help='Ratio of the validation set (default: 0.0)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N',
                    help='number of workers (default: 4)')
parser.add_argument('--model', type=str, default=None, required=True, metavar='MODEL',
                    help='model name (default: None)')
parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to resume training from (default: None)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--save_freq', type=int, default=25, metavar='N',
                    help='save frequency (default: 25)')
parser.add_argument('--eval_freq', type=int, default=5, metavar='N',
                    help='evaluation frequency (default: 5)')
parser.add_argument('--lr_init', type=float, default=0.01, metavar='LR',
                    help='initial learning rate (default: 0.01)')
parser.add_argument('--lr-type', type=str, default="wilson", metavar='S',
                    help='learning decay schedule type ("wilson" or "gupta" or "const")')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=1e-4,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--seed', type=int, default=200, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--log-distribution', action='store_true', default=False,
                    help='Whether to log distribution of weight and grad')
parser.add_argument('--log-error', action='store_true', default=False,
                    help='Whether to log quantization error of weight and grad')
for num in num_types:
    parser.add_argument('--wl-{}'.format(num), type=int, default=-1, metavar='N',
                        help='word length in bits for {}; -1 if full precision.'.format(num))
    parser.add_argument('--fl-{}'.format(num), type=int, default=-1, metavar='N',
                        help='number of fractional bits for {}; -1 if full precision.'.format(num))
    parser.add_argument('--{}-type'.format(num), type=str, default="block", metavar='S',
                        choices=["fixed", "block"],
                        help='quantization type for {}; fixed or block.'.format(num))
    parser.add_argument('--{}-rounding'.format(num), type=str, default='stochastic', metavar='S',
                        choices=["stochastic","nearest"],
                        help='rounding method for {}, stochastic or nearest'.format(num))
parser.add_argument('--no-quant-bias', action='store_true',
                    help='not quantize bias (default: off)')
parser.add_argument('--no-quant-bn', action='store_true',
                    help='not quantize batch norm (default: off)')
parser.add_argument('--auto_low', action='store_true', default=False,
                    help='auto_low')

args = parser.parse_args()

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

# Tensorboard Writer
log_name = os.path.join("./runs", args.name, "seed{}-{}".format(args.seed, time.time()))
print("Logging at {}".format(log_name))
writer = SummaryWriter(log_dir=log_name)


# Select quantizer
def quant_summary(number_type, wl, fl):
    if wl == -1:
        return "float"
    if number_type=="fixed":
        return "fixed-{}{}".format(wl, fl)
    elif number_type=="block":
        return "block-{}".format(wl)

for num in num_types:
    num_type = getattr(args, "{}_type".format(num))
    num_rounding = getattr(args, "{}_rounding".format(num))
    num_wl = getattr(args, "wl_{}".format(num))
    num_fl = getattr(args, "fl_{}".format(num))
    print("{}: {} rounding, {}".format(num, num_rounding,
          quant_summary(num_type, num_wl, num_fl)))

def make_quantizer(num):
    num_wl = getattr(args, "wl_{}".format(num))
    num_fl = getattr(args, "fl_{}".format(num))
    num_type = getattr(args, "{}_type".format(num))
    num_rounding = getattr(args, "{}_rounding".format(num))
    if num_type=="fixed":
        return lambda x : fixed_point_quantize(x, num_wl, num_fl, -1, -1, forward_rounding=num_rounding)
    elif num_type=="block":
        return lambda x : block_quantize(x, num_wl, -1, forward_rounding=num_rounding)

weight_quantizer = make_quantizer("weight")
grad_quantizer = make_quantizer("grad")
momentum_quantizer = make_quantizer("momentum")

dir_name = os.path.join("./checkpoint", args.name)
print('Preparing checkpoint directory {}'.format(dir_name))
os.makedirs(dir_name, exist_ok=True)
with open(os.path.join(dir_name, 'command.sh'), 'w') as f:
    f.write('python '+' '.join(sys.argv))
    f.write('\n')

loaders = get_data(args.dataset, args.data_path, args.batch_size, args.val_ratio, args.num_workers)

# Build model
print('Model: {}'.format(args.model))
model_cfg = getattr(models, args.model)
if 'LP' in args.model and args.wl_activate == -1 and args.wl_error == -1:
    raise Exception("Using low precision model but not quantizing activation or error")
elif 'LP' in args.model and (args.wl_activate != -1 or args.wl_error != -1):
    raise NotImplemented
    # model_cfg.kwargs.update(
    #     {"forward_wl":args.wl_activate, "forward_fl":args.fl_activate,
    #      "backward_wl":args.wl_error, "backward_fl":args.fl_error,
    #      "forward_layer_type":args.layer_type,
    #      "forward_round_type":args.quant_type})

if args.dataset=="CIFAR10": num_classes=10
elif args.dataset=="IMAGENET12": num_classes=1000
model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
model.cuda()
if args.auto_low:
    quant = lambda : Quantizer(args.wl_activate, args.wl_error,
                               args.fl_activate, args.fl_error,
                               args.activate_rounding, args.error_rounding,
                               args.activate_type, args.error_type)
    lower(model, quant, ["conv", "activation"])
print('SGD training')


def schedule(epoch, lr_schedule):
    if lr_schedule == "wilson":
        t = (epoch) / args.epochs
        lr_ratio = 0.01
        if t <= 0.5:
            factor = 1.0
        elif t <= 0.9:
            factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
        else:
            factor = lr_ratio
    elif lr_schedule == "const":
        factor = 1.0

    return args.lr_init * factor

criterion = F.cross_entropy
optimizer = SGDLP(
    model.parameters(),
    lr=args.lr_init,
    momentum=args.momentum,
    weight_decay=args.wd,
    weight_quant=weight_quantizer,
    grad_quant=grad_quantizer,
    momentum_quant=momentum_quantizer
)

start_epoch = 0
if args.resume is not None:
    print('Resume training from %s' % args.resume)
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint['epoch']-1
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

# Prepare logging
columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_loss', 'te_acc', 'time']

def log_result(writer, name, res, step):
    writer.add_scalar("{}/loss".format(name),     res['loss'],            step)
    writer.add_scalar("{}/acc_perc".format(name), res['accuracy'],        step)
    writer.add_scalar("{}/err_perc".format(name), 100. - res['accuracy'], step)

for epoch in range(start_epoch, args.epochs):
    time_ep = time.time()

    lr = schedule(epoch, args.lr_type)
    writer.add_scalar("lr", lr, epoch)
    utils.adjust_learning_rate(optimizer, lr)
    train_res = utils.train_epoch( loaders['train'], model, criterion,
            optimizer, weight_quantizer, grad_quantizer, writer, epoch,
            quant_bias=(not args.no_quant_bias),
            quant_bn=(not args.no_quant_bn),
            log_error=args.log_error)
    log_result(writer, "train", train_res, epoch+1)

    # Write parameters
    if args.log_distribution:
        for name, param in model.named_parameters():
            writer.add_histogram(
                "param/%s"%name, param.clone().cpu().data.numpy(), epoch)
            writer.add_histogram(
                "gradient/%s"%name, param.grad.clone().cpu().data.numpy(), epoch)

    # Validation
    if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1:
        utils.bn_update(loaders['train'], model)
        test_res = utils.eval(loaders['test'], model, criterion)
        log_result(writer, "test", test_res, epoch+1)
        if args.val_ratio > 0:
            val_res = utils.eval(loaders['val'], model, criterion)
            log_result(writer, "val", val_res, epoch+1)
    else:
        test_res = {'loss': None, 'accuracy': None}
        val_res = {'loss': None, 'accuracy': None}

    time_ep = time.time() - time_ep
    values = [epoch + 1, lr, train_res['loss'], train_res['accuracy'], test_res['loss'], test_res['accuracy'], time_ep]

    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
    if epoch % 40 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)

# Save the final model
if args.epochs % args.save_freq != 0:
    utils.save_checkpoint(
        dir_name,
        args.epochs,
        state_dict=model.state_dict(),
        optimizer=optimizer.state_dict()
    )
