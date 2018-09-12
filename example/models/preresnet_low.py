
"""
    PreResNet model definition
    ported from https://github.com/bearpaw/pytorch-classification/blob/master/models/cifar/preresnet.py
"""

import torch.nn as nn
import math
from qtorch import BlockQuantizer, FixedQuantizer

__all__ = ['PreResNet110LP', 'PreResNet20LP']


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, quant, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride
        self.quant = quant()

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.quant(out)
        out = self.conv1(out)
        out = self.quant(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.quant(out)
        out = self.conv2(out)
        out = self.quant(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, quant, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.quant = quant()

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.quant(out)
        out = self.conv1(out)
        out = self.quant(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.quant(out)
        out = self.conv2(out)
        out = self.quant(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.quant(out)
        out = self.conv3(out)
        out = self.quant(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class PreResNet(nn.Module):

    def __init__(self, forward_wl=-1, forward_fl=-1, backward_wl=-1, backward_fl=-1,
                 forward_layer_type="fixed", backward_layer_type="",
                 forward_round_type="stochastic", backward_round_type="",
                 num_classes=10, depth=110):

        assert forward_layer_type in ["block", "fixed"]
        assert forward_round_type in ["nearest", "stochastic"]
        if backward_layer_type == "": backward_layer_type = forward_layer_type
        if backward_round_type == "": backward_round_type = forward_round_type
        assert backward_layer_type in ["block", "fixed"]
        assert backward_round_type in ["nearest", "stochastic"]

        if forward_layer_type == "block":
            quant = lambda : BlockQuantizer(forward_wl, backward_wl, forward_round_type, backward_round_type)
        elif forward_layer_type == "fixed":
            quant = lambda : FixedQuantizer(forward_wl, forward_fl, backward_wl, backward_fl, forward_round_type, backward_round_type)

        super(PreResNet, self).__init__()
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        block = Bottleneck if depth >= 44 else BasicBlock

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.layer1 = self._make_layer(block, 16, n, quant)
        self.layer2 = self._make_layer(block, 32, n, quant, stride=2)
        self.layer3 = self._make_layer(block, 64, n, quant, stride=2)
        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.quant = quant()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, quant, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
            )

        layers = list()
        layers.append(block(self.inplanes, planes, quant , stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, quant))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.quant(x)

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = self.bn(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class PreResNet110LP:
    base = PreResNet
    args = list()
    kwargs = {'depth': 110}


class PreResNet20LP:
    base = PreResNet
    args = list()
    kwargs = {'depth': 20}
