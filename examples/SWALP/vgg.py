"""
    VGG model definition
    ported from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
"""

import math
import torch.nn as nn
from torch.nn.init import kaiming_normal_
import torchvision.transforms as transforms

__all__ = ["VGG16", "VGG16BN", "VGG19", "VGG19BN"]


def make_layers(cfg, batch_norm=False):
    layers = list()
    in_channels = 3
    n = 1
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            filters = int(v)
            conv2d = nn.Conv2d(in_channels, filters, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(filters), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            n += 1
            in_channels = filters
    return nn.Sequential(*layers)


def make_cls(hidden_dim):
    layers = []
    for i in range(2):
        layers += [
            nn.Dropout(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        ]
    return nn.Sequential(*layers)


cfg = {
    16: [
        "64",
        "64",
        "M",
        "128",
        "128",
        "M",
        "256",
        "256",
        "256",
        "M",
        "512",
        "512",
        "512",
        "M",
        "512",
        "512",
        "512",
        "M",
    ],
    19: [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGG(nn.Module):
    def __init__(self, num_classes=10, depth=16, batch_norm=False):

        super(VGG, self).__init__()
        self.features = make_layers(cfg[depth], batch_norm)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Base:
    base = VGG
    args = list()
    kwargs = dict()
    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )


class VGG16(Base):
    pass


class VGG16BN(Base):
    kwargs = {"batch_norm": True}


class VGG19(Base):
    kwargs = {"depth": 19}


class VGG19BN(Base):
    kwargs = {"depth": 19, "batch_norm": True}
