import argparse
import time
import torch
import torch.nn.functional as F
import utils
import tabulate
import vgg
from qtorch.quant import *
from qtorch.optim import OptimLP
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from qtorch import BlockFloatingPoint, FixedPoint, FloatingPoint
from qtorch.auto_low import sequential_lower
import torchvision.models as models

num_types = ["weight", "activate", "grad", "error", "momentum"]

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
    "--swa_start", type=int, default=200, metavar="N", help="SWALP start epoch"
)
parser.add_argument(
    "--swa_lr",
    type=float,
    default=0.01,
    metavar="LR",
    help="SWALP learning rate (default: 0.01)",
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
    help="number of epochs to train (default: 300)",
)
parser.add_argument(
    "--lr_init",
    type=float,
    default=0.05,
    metavar="LR",
    help="initial learning rate (default: 0.01)",
)
parser.add_argument(
    "--wd", type=float, default=1e-4, help="weight decay (default: 1e-4)"
)
parser.add_argument(
    "--seed", type=int, default=1, metavar="N", help="random seed (default: 1)"
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
    "--rounding".format(num),
    type=str,
    default="stochastic",
    metavar="S",
    choices=["stochastic", "nearest"],
    help="rounding method for {}, stochastic or nearest".format(num),
)

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
utils.set_seed(args.seed, args.cuda)

loaders = utils.get_data(args.dataset, args.data_path, args.batch_size, num_workers=8)
num_classes = utils.num_classes_dict[args.dataset]

# prepare quantization functions
# using block floating point, allocating shared exponent along the first dimension
number_dict = dict()
for num in num_types:
    num_wl = getattr(args, "wl_{}".format(num))
    number_dict[num] = BlockFloatingPoint(wl=num_wl, dim=0)
    print("{:10}: {}".format(num, number_dict[num]))
quant_dict = dict()
for num in ["weight", "momentum", "grad"]:
    quant_dict[num] = quantizer(
        forward_number=number_dict[num], forward_rounding=args.rounding
    )

# Build model
print("Base Model: {}".format(args.model))
model_cfg = getattr(vgg, args.model)
model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)

# automatically insert quantization modules
model = sequential_lower(
    model,
    layer_types=["conv", "linear"],
    forward_number=number_dict["activate"],
    backward_number=number_dict["error"],
    forward_rounding=args.rounding,
    backward_rounding=args.rounding,
)
model.classifier[-1] = model.classifier[-1][0]  # removing the final quantization module
model.cuda()

# Build SWALP model
swa_model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
swa_model.swa_n = 0
swa_model.cuda()

criterion = F.cross_entropy
optimizer = SGD(model.parameters(), lr=args.lr_init, momentum=0.9, weight_decay=args.wd)
# insert quantizations into the optimization loops
optimizer = OptimLP(
    optimizer,
    weight_quant=quant_dict["weight"],
    grad_quant=quant_dict["grad"],
    momentum_quant=quant_dict["momentum"],
)


def schedule(epoch):
    if epoch < args.swa_start:
        t = (epoch) / args.swa_start
        lr_ratio = 0.01
        if t <= 0.5:
            factor = 1.0
        elif t <= 0.9:
            factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
        else:
            factor = lr_ratio
        return factor
    else:
        return args.swa_lr / args.lr_init


# learning rate schedule
scheduler = LambdaLR(optimizer, lr_lambda=[schedule])


# Prepare logging
columns = [
    "ep",
    "lr",
    "tr_loss",
    "tr_acc",
    "tr_time",
    "te_loss",
    "te_acc",
    "swa_te_loss",
    "swa_te",
]

for epoch in range(args.epochs):
    # lr = utils.schedule(epoch, args.lr_init, args.swa_start, args.swa_lr)
    # utils.adjust_learning_rate(optimizer, lr)
    scheduler.step()
    train_res = utils.run_epoch(
        loaders["train"], model, criterion, optimizer=optimizer, phase="train"
    )
    test_res = utils.run_epoch(loaders["test"], model, criterion, phase="eval")

    if epoch >= args.swa_start:
        utils.moving_average(swa_model, model, 1.0 / (swa_model.swa_n + 1))
        swa_model.swa_n += 1
        swa_te_res = utils.run_epoch(
            loaders["test"], swa_model, criterion, phase="eval"
        )
    else:
        swa_te_res = {"loss": None, "accuracy": None}

    values = [
        epoch + 1,
        optimizer.param_groups[0]["lr"],
        *train_res.values(),
        *test_res.values(),
        *swa_te_res.values(),
    ]
    utils.print_table(columns, values, epoch)
