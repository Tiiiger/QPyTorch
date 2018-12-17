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
from data import get_data
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler
from qtorch.quant import *
from qtorch.auto_low import lower
from qtorch.optim import SGDLP
from torch.optim import SGD
from qtorch import BlockFloatingPoint, FixedPoint, FloatingPoint

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
    parser.add_argument('--{}-man'.format(num), type=int, default=-1, metavar='N',
                        help='number of bits to use for mantissa of {}; -1 if full precision.'.format(num))
    parser.add_argument('--{}-exp'.format(num), type=int, default=-1, metavar='N',
                        help='number of bits to use for exponent of {}; -1 if full precision.'.format(num))
    parser.add_argument('--{}-type'.format(num), type=str, default="block", metavar='S',
                        choices=["fixed", "block", "float", "full"],
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
parser.add_argument('--float', action='store_true', default=False,
                    help='use single precision model')
parser.add_argument('--half', action='store_true', default=False,
                    help='use half precision model')

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
def quant_summary(number_type, wl=-1, fl=-1, man=-1, exp=-1):
    if wl == -1 and man == -1:
        return "native float"
    if number_type=="fixed":
        return "fixed-{}{}".format(wl, fl)
    elif number_type=="block":
        return "block-{}".format(wl)
    elif number_type=="float":
        return "float-e{}-m{}".format(man, exp)

for num in num_types:
    num_type = getattr(args, "{}_type".format(num))
    num_rounding = getattr(args, "{}_rounding".format(num))
    num_wl = getattr(args, "wl_{}".format(num))
    num_fl = getattr(args, "fl_{}".format(num))
    num_man = getattr(args, "{}_man".format(num))
    num_exp = getattr(args, "{}_exp".format(num))
    print("{}: {} rounding, {}".format(num, num_rounding,
          quant_summary(num_type, wl=num_wl, fl=num_fl, man=num_man, exp=num_exp)))

def make_number(number, wl=-1, fl=-1, exp=-1, man=-1):
    if number == "fixed":
        return FixedPoint(wl, fl)
    elif number == "block":
        return BlockFloatingPoint(wl)
    elif number == "float":
        return FloatingPoint(exp, man)
    else:
        raise ValueError("Not supported number type")

def make_quantizer(num):
    num_wl = getattr(args, "wl_{}".format(num))
    num_fl = getattr(args, "fl_{}".format(num))
    num_type = getattr(args, "{}_type".format(num))
    num_rounding = getattr(args, "{}_rounding".format(num))
    num_man = getattr(args, "{}_man".format(num))
    num_exp = getattr(args, "{}_exp".format(num))
    if num_type == "full": return lambda x : x
    forward_number = make_number(num_type, wl=num_wl, fl=num_fl, exp=num_exp, man=num_man)
    backward_number = make_number(num_type, wl=num_wl, fl=num_fl, exp=num_exp, man=num_man)
    return Quantizer(
               forward_number, backward_number,
               num_rounding, num_rounding)

if not args.float:
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
    activate_number = make_number(args.activate_type, wl=args.wl_activate, fl=args.fl_activate,
                                  exp=args.activate_exp, man=args.activate_man)
    error_number = make_number(args.error_type, wl=args.wl_error, fl=args.fl_error,
                               exp=args.error_exp, man=args.error_man)
    make_quant = lambda : Quantizer(activate_number, error_number, args.activate_rounding, args.error_rounding)
    model_cfg.kwargs.update({"quant":make_quant})

if args.dataset=="CIFAR10": num_classes=10
elif args.dataset=="IMAGENET12": num_classes=1000
model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
model.cuda()
if args.auto_low:
    lower(model,
          layer_types=["activation"],
          forward_number=make_number(
                             args.activate_type,
                             wl=args.wl_activate,
                             fl=args.fl_activate,
                             man=args.activate_man,
                             exp=args.activate_exp,
                         ),
          backward_number=make_number(
                              args.error_type,
                              wl=args.wl_error,
                              fl=args.fl_error,
                              man=args.error_man,
                              exp=args.error_exp,
                          ),
          forward_rounding=args.activate_rounding,
          backward_rounding=args.error_rounding
    )
if args.half:
    model.half()
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
optimizer = SGD(
   model.parameters(),
   lr=args.lr_init,
   momentum=args.momentum,
   weight_decay=args.wd,
)
if not args.float:
    optimizer = SGDLP(optimizer,
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
columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'tr_time', 'te_loss', 'te_acc', 'te_time']

def log_result(writer, name, res, step):
    writer.add_scalar("{}/loss".format(name),     res['loss'],            step)
    writer.add_scalar("{}/acc_perc".format(name), res['accuracy'],        step)
    writer.add_scalar("{}/err_perc".format(name), 100. - res['accuracy'], step)
    writer.add_scalar("{}/time_pass".format(name), res['time_pass'], step)

for epoch in range(start_epoch, args.epochs):
    time_ep = time.time()

    lr = schedule(epoch, args.lr_type)
    writer.add_scalar("lr", lr, epoch)
    utils.adjust_learning_rate(optimizer, lr)
    train_res = utils.run_epoch(loaders['train'], model, criterion,
                                optimizer=optimizer, writer=writer,
                                log_error=args.log_error, phase="train",
                                half=args.half)
    time_pass = time.time() - time_ep
    train_res['time_pass'] = time_pass
    log_result(writer, "train", train_res, epoch+1)

    # Write parameters
    if args.log_distribution:
        for name, param in model.named_parameters():
            writer.add_histogram(
                "param/%s"%name, param.clone().cpu().data.numpy(), epoch)
            writer.add_histogram(
                "gradient/%s"%name, param.grad.clone().cpu().data.numpy(), epoch)

    # Validation
    if args.val_ratio > 0:
        time_ep = time.time()
        val_res = utils.eval(loaders['val'], model, criterion)
        time_pass = time.time() - time_ep
        val_res['time_pass'] = time_pass
        log_result(writer, "val", val_res, epoch+1)

    if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1:
        # utils.bn_update(loaders['train'], model)
        time_ep = time.time()
        test_res = utils.run_epoch(loaders['test'], model, criterion, phase="eval", half=args.half)
        time_pass = time.time() - time_ep
        test_res['time_pass'] = time_pass
        log_result(writer, "test", test_res, epoch+1)

        if args.val_ratio > 0:
            time_ep = time.time()
            val_res = utils.run_epoch(loaders['val'], model, criterion, phase="eval", half=args.half)
            time_pass = time.time() - time_ep
            val_res['time_pass'] = time_pass
            log_result(writer, "val", val_res, epoch+1)
    else:
        test_res = {'loss': None, 'accuracy': None, 'time_pass': None}

    values = [epoch + 1, lr, train_res['loss'], train_res['accuracy'], train_res['time_pass'], test_res['loss'], test_res['accuracy'], test_res['time_pass']]

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
