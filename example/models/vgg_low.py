"""
    VGG model definition
    ported from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
"""

import math
import torch.nn as nn
import torchvision.transforms as transforms
from qtorch.quant import Quantizer

__all__ = ['VGG16LP', 'VGG16BNLP', 'VGG19LP', 'VGG19BNLP']


def make_layers(cfg, quant, batch_norm=False):
    layers = list()
    in_channels = 3
    n = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            use_quant = v[-1] != 'N'
            filters = int(v) if use_quant else int(v[:-1])
            conv2d = nn.Conv2d(in_channels, filters, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(filters), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            if use_quant: layers += [quant()]
            n += 1
            in_channels = filters
    return nn.Sequential(*layers)


cfg = {
    16: ['64', '64', 'M', '128', '128', 'M', '256', '256', '256', 'M', '512', '512', '512', 'M', '512', '512', '512', 'M'],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
         512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, forward_wl=-1, forward_fl=-1, backward_wl=-1, backward_fl=-1,
                 forward_layer_type="stochastic", backward_layer_type="stochastic",
                 forward_round_type="fixed", backward_round_type="fixed",
                 num_classes=10, depth=16, batch_norm=False):

        #assert forward_layer_type in ["block", "fixed"]
        #assert forward_round_type in ["nearest", "stochastic"]
        if backward_layer_type == "": backward_layer_type = forward_layer_type
        if backward_round_type == "": backward_round_type = forward_round_type
        #assert backward_layer_type in ["block", "fixed"]
        #assert backward_round_type in ["nearest", "stochastic"]

        quant = lambda : Quantizer(16, 16, 16, 16, "nearest", "nearest", "fixed", "fixed")

        super(VGG, self).__init__()
        self.features = make_layers(cfg[depth], quant, batch_norm)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            quant(),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            quant(),
            nn.Linear(512, num_classes),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
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
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


class VGG16LP(Base):
    pass


class VGG16BNLP(Base):
    kwargs = {'batch_norm': True}


class VGG19LP(Base):
    kwargs = {'depth': 19}


class VGG19BNLP(Base):
    kwargs = {'depth': 19, 'batch_norm': True}
