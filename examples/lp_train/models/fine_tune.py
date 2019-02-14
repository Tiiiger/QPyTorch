"""
    low precision Alex net definition
    reference: https://arxiv.org/abs/1511.06393
"""
import torch.nn as nn
import torchvision.transforms as transforms
import math
from .quantizer import BlockQuantizer, FixedQuantizer

__all__ = ["FineTuneNet"]

class CNN(nn.Module):
    def __init__(self, wl_activate=-1, fl_activate=-1, wl_error=-1, fl_error=-1,
            quantize_type="fixed", quantize_backward=False, num_classes=10, writer=None):
        if quantize_type == "block":
            quant = lambda : BlockQuantizer(wl_activate, wl_error, "stochastic", quantize_backward)
        elif quantize_type == "fixed":
            quant = lambda : FixedQuantizer(wl_activate, fl_activate, "stochastic", quantize_backward)
        super(CNN, self).__init__()
        layers = []
        cfg =  [256, 128, 'M', 256, 256]
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True), quant()]
                in_channels = v
        layers+=[nn.Conv2d(256, 128, kernel_size=7), nn.ReLU(inplace=True), quant(), nn.MaxPool2d(kernel_size=2, stride=2)]
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(128*5*5, 10)
        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
        #        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class Base:
    base = CNN
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

class FineTuneNet(Base):
    pass
