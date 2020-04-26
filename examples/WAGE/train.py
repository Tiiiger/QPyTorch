import argparse
import os
import sys
import time
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import utils
import tabulate
import models
import wage_qtorch
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler

num_types = ["weight", "activate", "grad", "error", "momentum"]

parser = argparse.ArgumentParser(description="SGD/SWA training")
parser.add_argument(
    "--dir",
    type=str,
    default=None,
    required=True,
    help="training directory (default: None)",
)
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
    "--epochs",
    type=int,
    default=300,
    metavar="N",
    help="number of epochs to train (default: 200)",
)
parser.add_argument(
    "--seed", type=int, default=200, metavar="N", help="random seed (default: 1)"
)
for num in num_types:
    parser.add_argument(
        "--wl-{}".format(num),
        type=int,
        default=-1,
        metavar="N",
        help="word length in bits for {}; -1 if full precision.".format(num),
    )
    parser.add_argument(
        "--fl-{}".format(num),
        type=int,
        default=-1,
        metavar="N",
        help="number of fractional bits for {}; -1 if full precision.".format(num),
    )
    parser.add_argument(
        "--{}-rounding".format(num),
        type=str,
        default="stochastic",
        metavar="S",
        choices=["stochastic", "nearest"],
        help="rounding method for {}, stochastic or nearest".format(num),
    )
parser.add_argument(
    "--wl-rand",
    type=int,
    default=-1,
    metavar="N",
    help="word length in bits for rand number; -1 if full precision.",
)
parser.add_argument(
    "--qtorch",
    action="store_true",
    default=False,
    help="Whether to use qtorch quantization.",
)

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
utils.set_seed(args.seed, args.cuda)

weight_quantizer = lambda x, scale: wage_qtorch.QW(
    x, args.wl_weight, scale, mode=args.weight_rounding
)
grad_clip = lambda x: wage_qtorch.C(x, args.wl_weight)
if args.wl_weight == -1:
    weight_quantizer = None
if args.wl_grad == -1:
    grad_quantizer = None

assert args.dataset in ["CIFAR10"]
print("Loading dataset {} from {}".format(args.dataset, args.data_path))

ds = getattr(datasets, args.dataset)
path = os.path.join(args.data_path, args.dataset.lower())
transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)
transform_test = transforms.Compose([transforms.ToTensor(),])
train_set = ds(path, train=True, download=True, transform=transform_train)
test_set = ds(path, train=False, download=True, transform=transform_test)
num_classes = 10

loaders = {
    "train": torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    ),
    "test": torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    ),
}

# Build model
print("Model: {}".format(args.model))
model_cfg = getattr(models, args.model)
from functools import partial

m = partial(
    wage_qtorch.WAGEQuantizer, A_mode=args.activate_rounding, E_mode=args.error_rounding
)
model_cfg.kwargs.update(
    {
        "quantizer": m,
        "wl_activate": args.wl_activate,
        "wl_error": args.wl_error,
        "wl_weight": args.wl_weight,
    }
)

model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
if args.cuda:
    model.cuda()
    for name, param_acc in model.weight_acc.items():
        model.weight_acc[name] = param_acc.cuda()

criterion = utils.SSE


def schedule(epoch):
    if epoch < 200:
        return 8.0
    elif epoch < 250:
        return 1
    else:
        return 1 / 8.0


start_epoch = 0

# Prepare logging
columns = ["ep", "lr", "tr_loss", "tr_acc", "te_loss", "te_acc", "time"]

for epoch in range(start_epoch, args.epochs):
    time_ep = time.time()
    lr = schedule(epoch)
    grad_quantizer = lambda x: wage_qtorch.QG(
        x, args.wl_grad, args.wl_rand, lr, mode=args.grad_rounding
    )

    train_res = utils.train_epoch(
        loaders["train"],
        model,
        criterion,
        weight_quantizer,
        grad_quantizer,
        epoch,
        wage_quantize=True,
        wage_grad_clip=grad_clip,
    )

    # Validation
    test_res = utils.eval(loaders["test"], model, criterion, weight_quantizer)

    time_ep = time.time() - time_ep
    values = [
        epoch + 1,
        lr,
        train_res["loss"],
        train_res["accuracy"],
        test_res["loss"],
        test_res["accuracy"],
        time_ep,
    ]

    table = tabulate.tabulate([values], columns, tablefmt="simple", floatfmt="8.4f")
    if epoch % 20 == 0:
        table = table.split("\n")
        table = "\n".join([table[1]] + table)
    else:
        table = table.split("\n")[2]
    print(table)
