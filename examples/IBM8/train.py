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
from qtorch.auto_low import lower
from qtorch.optim import OptimLP
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import SGD
from qtorch import BlockFloatingPoint, FixedPoint, FloatingPoint
from qtorch.quant import quantizer, Quantizer

num_types = ["weight", "activate", "grad", "error", "momentum", "acc"]

parser = argparse.ArgumentParser(description="SGD/SWA training")
parser.add_argument(
    "--dataset", type=str, default="CIFAR10", help="dataset name: CIFAR10 or IMAGENET12"
)
parser.add_argument(
    "--data_path",
    type=str,
    default="./data",
    required=True,
    metavar="PATH",
    help='path to datasets location (default: "./data")',
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=128,
    metavar="N",
    help="input batch size (default: 128)",
)
parser.add_argument(
    "--val_ratio",
    type=float,
    default=0.0,
    metavar="N",
    help="Ratio of the validation set (default: 0.0)",
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=4,
    metavar="N",
    help="number of workers (default: 4)",
)
parser.add_argument(
    "--model",
    type=str,
    default=None,
    required=True,
    metavar="MODEL",
    help="model name (default: None)",
)
parser.add_argument(
    "--resume",
    type=str,
    default=None,
    metavar="CKPT",
    help="checkpoint to resume training from (default: None)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=200,
    metavar="N",
    help="number of epochs to train (default: 200)",
)
parser.add_argument(
    "--lr_init",
    type=float,
    default=0.01,
    metavar="LR",
    help="initial learning rate (default: 0.01)",
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.9,
    metavar="M",
    help="SGD momentum (default: 0.9)",
)
parser.add_argument(
    "--wd", type=float, default=1e-4, help="weight decay (default: 1e-4)"
)
for num in num_types:
    parser.add_argument(
        "--{}-man".format(num),
        type=int,
        default=-1,
        metavar="N",
        help="number of bits to use for mantissa of {}; -1 if full precision.".format(
            num
        ),
    )
    parser.add_argument(
        "--{}-exp".format(num),
        type=int,
        default=-1,
        metavar="N",
        help="number of bits to use for exponent of {}; -1 if full precision.".format(
            num
        ),
    )
    parser.add_argument(
        "--{}-rounding".format(num),
        type=str,
        default="stochastic",
        metavar="S",
        choices=["stochastic", "nearest"],
        help="rounding method for {}, stochastic or nearest".format(num),
    )

args = parser.parse_args()

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# Select quantizer
def quant_summary(man=-1, exp=-1):
    return "float-e{}-m{}".format(man, exp)


loaders = get_data(
    args.dataset, args.data_path, args.batch_size, args.val_ratio, args.num_workers
)
if args.dataset == "CIFAR10":
    num_classes = 10
elif args.dataset == "IMAGENET12":
    num_classes = 1000

quantizers = {}
for num in num_types:
    num_rounding = getattr(args, "{}_rounding".format(num))
    num_man = getattr(args, "{}_man".format(num))
    num_exp = getattr(args, "{}_exp".format(num))
    number = FloatingPoint(exp=num_exp, man=num_man)
    print("{}: {} rounding, {}".format(num, num_rounding, number))
    quantizers[num] = quantizer(forward_number=number, forward_rounding=num_rounding)
# Build model
print("Model: {}".format(args.model))
model_cfg = getattr(models, args.model)
if "LP" in args.model:
    activate_number = FloatingPoint(exp=args.activate_exp, man=args.activate_man)
    error_number = FloatingPoint(exp=args.error_exp, man=args.error_man)
    print("activation: {}, {}".format(args.activate_rounding, activate_number))
    print("error: {}, {}".format(args.error_rounding, error_number))
    make_quant = lambda: Quantizer(
        activate_number, error_number, args.activate_rounding, args.error_rounding
    )
    model_cfg.kwargs.update({"quant": make_quant})

model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
model.cuda()

criterion = F.cross_entropy
optimizer = SGD(
    model.parameters(), lr=args.lr_init, momentum=args.momentum, weight_decay=args.wd,
)
optimizer = OptimLP(
    optimizer,
    weight_quant=quantizers["weight"],
    grad_quant=quantizers["grad"],
    momentum_quant=quantizers["momentum"],
    acc_quant=quantizers["acc"],
    grad_scaling=1 / 1000,  # scaling down the gradient
)


def schedule(epoch):
    t = (epoch) / args.epochs
    lr_ratio = 0.01
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio

    return factor


scheduler = LambdaLR(optimizer, lr_lambda=[schedule])
# Prepare logging
columns = ["ep", "lr", "tr_loss", "tr_acc", "tr_time", "te_loss", "te_acc", "te_time"]


def get_result(loaders, model, phase):
    time_ep = time.time()
    res = utils.run_epoch(
        loaders[phase], model, criterion, optimizer=optimizer, phase=phase
    )
    time_pass = time.time() - time_ep
    res["time_pass"] = time_pass
    return res


for epoch in range(args.epochs):

    scheduler.step()
    time_ep = time.time()
    train_res = get_result(loaders, model, "train")

    if epoch == 0 or epoch % 5 == 4 or epoch == args.epochs - 1:
        test_res = get_result(loaders, model, "test")
    else:
        test_res = {"loss": None, "accuracy": None, "time_pass": None}

    values = [
        epoch + 1,
        optimizer.param_groups[0]["lr"],
        train_res["loss"],
        train_res["accuracy"],
        train_res["time_pass"],
        test_res["loss"],
        test_res["accuracy"],
        test_res["time_pass"],
    ]

    utils.print_table(values, columns, epoch)
