import argparse
import os
import sys
import time
import json
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import utils
import tabulate
import models
from qtorch import block_quantize, fixed_point_quantize
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler
from auto_low import resnet_lower, lower

parser = argparse.ArgumentParser(description='SGD/SWA training')
parser.add_argument('--dir', type=str, default=None, required=True,
                    help='training directory (default: None)')
parser.add_argument('--dataset', type=str, default='CIFAR10',
                    help='dataset name: CIFAR10 or IMAGENET12')
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
parser.add_argument('--swa', action='store_true',
                    help='swa usage flag (default: off)')
parser.add_argument('--swa_start', type=float, default=161, metavar='N',
                    help='SWA start epoch number (default: 161)')
parser.add_argument('--swa_lr', type=float, default=0.05, metavar='LR',
                    help='SWA LR (default: 0.05)')
parser.add_argument('--swa_c_epochs', type=int, default=1, metavar='N',
                    help='SWA model collection frequency/cycle length in epochs (default: 1)')
parser.add_argument('--seed', type=int, default=200, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--log-name', type=str, default='', metavar='S',
                    help="Name for the log dir")
parser.add_argument('--log-distribution', action='store_true', default=False,
                    help='Whether to log distribution of weight and grad')
parser.add_argument('--log-error', action='store_true', default=False,
                    help='Whether to log quantization error of weight and grad')
parser.add_argument('--wl-weight', type=int, default=-1, metavar='N',
                    help='word length in bits for weight output; -1 if full precision.')
parser.add_argument('--fl-weight', type=int, default=-1, metavar='N',
                    help='float length in bits for weight output; -1 if full precision.')
parser.add_argument('--wl-grad', type=int, default=-1, metavar='N',
                    help='word length in bits for gradient; -1 if full precision.')
parser.add_argument('--fl-grad', type=int, default=-1, metavar='N',
                    help='float length in bits for gradient; -1 if full precision.')
parser.add_argument('--wl-activate', type=int, default=-1, metavar='N',
                    help='word length in bits for layer activattions; -1 if full precision.')
parser.add_argument('--fl-activate', type=int, default=-1, metavar='N',
                    help='float length in bits for layer activations; -1 if full precision.')
parser.add_argument('--wl-error', type=int, default=-1, metavar='N',
                    help='word length in bits for backward error; -1 if full precision.')
parser.add_argument('--fl-error', type=int, default=-1, metavar='N',
                    help='float length in bits for backward error; -1 if full precision.')
parser.add_argument('--weight-type', type=str, default="fixed", metavar='S',
                    help='quantization type for weight; fixed or block.')
parser.add_argument('--grad-type', type=str, default="fixed", metavar='S',
                    help='quantization type for gradient; fixed or block.')
parser.add_argument('--layer-type', type=str, default="fixed", metavar='S',
                    help='quantization type for layer activation and error; fixed or block.')
parser.add_argument('--quant-type', type=str, default='stochastic', metavar='S',
                    help='rounding method, stochastic or nearest ')
parser.add_argument('--quant-backward', action='store_true', default=False,
                    help='not quantize backward (default: off)')
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
if args.log_name != "":
    log_name = "{}-{}-seed{}-{}".format( args.log_name, "swa" if args.swa else "sgd", args.seed, int(time.time()))
else:
    log_name = "{}-seed{}-{}".format("swa" if args.swa else "sgd", args.seed, int(time.time()))
print("Logging at {}".format(log_name))
writer = SummaryWriter(log_dir=os.path.join(".", "runs", log_name)) 


# Select quantizer
for i in [args.weight_type, args.grad_type, args.layer_type]:
    assert i in ["fixed", "block"]
def quant_summary(number_type, wl, fl):
    if wl == -1:
        return "float"
    if number_type=="fixed":
        return "fixed-{}{}".format(wl, fl)
    elif number_type=="block":
        return "block-{}".format(wl)

w_summary = quant_summary(args.weight_type, args.wl_weight, args.fl_weight)
g_summary = quant_summary(args.grad_type, args.wl_grad, args.fl_grad)
a_summary = quant_summary(args.layer_type, args.wl_activate, args.fl_activate)
e_summary = quant_summary(args.layer_type, args.wl_error, args.fl_error)
print("{} rounding, W:{}, A:{}, G:{}, E:{}".format(args.quant_type, w_summary,
                                                   a_summary, g_summary,
                                                   e_summary))

def make_quantizer(number_type, wl, fl, quant_type):
    if number_type=="fixed":
        return lambda x : fixed_point_quantize(x, wl, fl, -1, -1, quant_type)
    elif number_type=="block":
        return lambda x : block_quantize(x, wl, -1, quant_type)

weight_quantizer = make_quantizer(args.weight_type, args.wl_weight,
                                  args.fl_weight, args.quant_type)
grad_quantizer = make_quantizer(args.grad_type, args.wl_grad,
                                  args.fl_grad, args.quant_type)

dir_name = args.dir + "-seed-{}".format(args.seed)
print('Preparing checkpoint directory {}'.format(dir_name))
os.makedirs(dir_name, exist_ok=True)
with open(os.path.join(dir_name, 'command.sh'), 'w') as f:
    f.write('python '+' '.join(sys.argv))
    f.write('\n')

assert args.dataset in ["CIFAR10", "IMAGENET12"]
print('Loading dataset {} from {}'.format(args.dataset, args.data_path))
if args.dataset=="CIFAR10":
    ds = getattr(datasets, args.dataset)
    path = os.path.join(args.data_path, args.dataset.lower())
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_set = ds(path, train=True, download=True, transform=transform_train)
    val_set = ds(path, train=True, download=True, transform=transform_test)
    test_set = ds(path, train=False, download=True, transform=transform_test)
    if args.val_ratio != 0:
        train_size = len(train_set)
        indices = list(range(train_size))
        val_size = int(args.val_ratio*train_size)
        print("train set size {}, validation set size {}".format(train_size-val_size, val_size))
        np.random.shuffle(indices)
        val_idx, train_idx = indices[train_size-val_size:], indices[:train_size-val_size]
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
    else :
        train_sampler = None
        val_sampler = None
    num_classes = 10
elif args.dataset=="IMAGENET12":
    traindir = os.path.join(args.data_path, args.dataset.lower(), 'train')
    valdir = os.path.join(args.data_path, args.dataset.lower(), 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_set = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    test_set = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    train_sampler = None
    val_sampler = None
    num_classes = 1000


loaders = {
    'train': torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    ),
    'val': torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    ),
    'test': torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
}

# Build model 
print('Model: {}'.format(args.model))
model_cfg = getattr(models, args.model)
if 'LP' in args.model and args.wl_activate == -1 and args.wl_error == -1:
    raise Exception("Using low precision model but not quantizing activation or error")
elif 'LP' in args.model and (args.wl_activate != -1 or args.wl_error != -1):
    model_cfg.kwargs.update(
        {"forward_wl":args.wl_activate, "forward_fl":args.fl_activate,
         "backward_wl":args.wl_error, "backward_fl":args.fl_error,
         "forward_layer_type":args.layer_type,
         "forward_round_type":args.quant_type})

model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
model.cuda()
if args.auto_low:
    if args.layer_type == "block":
        quant = lambda : BlockQuantizer(args.wl_activate, args.wl_error, args.quant_type, args.quant_type)
    elif args.layer_type == "fixed":
        quant = lambda : FixedQuantizer(args.wl_activate, args.wl_activate, args.wl_error, args.fl_error, args.quant_type, args.quant_type)
    lower(model, quant, ["conv","activation"])
if args.swa:
    print('SWA training')
    if 'LP' in args.model:
        model_cfg.kwargs = {}
    swa_model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
    swa_model.cuda()
    swa_n = 0
else:
    print('SGD training')


def schedule(epoch, lr_schedule):
    if lr_schedule == "wilson":
        t = (epoch) / (args.swa_start if args.swa else args.epochs)
        lr_ratio = args.swa_lr / args.lr_init if args.swa else 0.01
        if t <= 0.5:
            factor = 1.0
        elif t <= 0.9:
            factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
        else:
            factor = lr_ratio
    elif lr_schedule == "gupta":
        if epoch < 50: factor = 1
        elif 50 <= epoch < 70: factor = 0.5**1
        elif 70 <= epoch < 100: factor = 0.5**2
        elif 100 <= epoch: factor = 0.5**3
    elif lr_schedule == "const":
        factor = 1.0

    return args.lr_init * factor

criterion = F.cross_entropy
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=args.lr_init,
    momentum=args.momentum,
    weight_decay=args.wd
)

start_epoch = 0
if args.resume is not None:
    print('Resume training from %s' % args.resume)
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint['epoch']-1
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if args.swa:
        swa_state_dict = checkpoint['swa_state_dict']
        if swa_state_dict is not None:
            swa_model.load_state_dict(swa_state_dict)
        swa_n_ckpt = checkpoint['swa_n']
        if swa_n_ckpt is not None:
            swa_n = swa_n_ckpt

# Prepare logging
columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_loss', 'te_acc', 'time']
if args.swa:
    columns = columns[:-1] + ['swa_te_loss', 'swa_te_acc'] + columns[-1:]
    swa_res = {'loss': None, 'accuracy': None}

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

    if args.swa and (epoch + 1) >= args.swa_start and (epoch + 1 - args.swa_start) % args.swa_c_epochs == 0:
        decay = 1.0 / (swa_n+1)
        utils.moving_average(swa_model, model, decay)
        swa_n += 1
        if args.log_distribution:
            for name, param in swa_model.named_parameters():
                writer.add_histogram( "param-swa/%s"%name,
                    param.clone().cpu().data.numpy(), epoch)
        if epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1:
            utils.bn_update(loaders['train'], swa_model)
            swa_te_res = utils.eval(loaders['test'], swa_model, criterion)
            log_result(writer, "test_avged", swa_te_res, epoch+1)
            if args.val_ratio > 0 :
                swa_val_res = utils.eval(loaders['val'], swa_model, criterion)
                log_result(writer, "val_avged", swa_val_res, epoch+1)
        else:
            swa_te_res = {'loss': None, 'accuracy': None}
            swa_val_res = {'loss': None, 'accuracy': None}
    else:
       swa_te_res = {'loss': None, 'accuracy': None}
       swa_val_res = {'loss': None, 'accuracy': None}

    # Save Checkpoint

    if (epoch+1) % args.save_freq == 0 or (epoch+1) == args.swa_start:
        utils.save_checkpoint(
            dir_name,
            epoch + 1,
            state_dict=model.state_dict(),
            swa_state_dict=swa_model.state_dict() if args.swa else None,
            swa_n=swa_n if args.swa else None,
            optimizer=optimizer.state_dict()
        )

    time_ep = time.time() - time_ep
    values = [epoch + 1, lr, train_res['loss'], train_res['accuracy'], test_res['loss'], test_res['accuracy'], time_ep]

    if args.swa :
        values = values[:-1] + [swa_te_res['loss'], swa_te_res['accuracy']] + values[-1:]

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
        swa_state_dict=swa_model.state_dict() if args.swa else None,
        swa_n=swa_n if args.swa else None,
        optimizer=optimizer.state_dict()
    )


