import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from qtorch import BlockQuantizer, FixedQuantizer

__all__ = ["DenseNetBC100LP"]


def _bn_function_factory(norm, relu, conv, quant):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = quant(conv(relu(quant(norm(concated_features)))))
        return bottleneck_output

    return bn_function


class _DenseLayer(nn.Module):
    def __init__(
        self,
        num_input_features,
        growth_rate,
        bn_size,
        drop_rate,
        quant,
        efficient=False,
    ):
        super(_DenseLayer, self).__init__()
        self.add_module("norm1", nn.BatchNorm2d(num_input_features))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module(
            "conv1",
            nn.Conv2d(
                num_input_features,
                bn_size * growth_rate,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )
        self.add_module("norm2", nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.add_module(
            "conv2",
            nn.Conv2d(
                bn_size * growth_rate,
                growth_rate,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )
        self.quant = quant()
        self.drop_rate = drop_rate
        self.efficient = efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(
            self.norm1, self.relu1, self.conv1, self.quant
        )
        if self.efficient and any(
            prev_feature.requires_grad for prev_feature in prev_features
        ):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.quant(
            self.conv2(self.relu2(self.quant(self.norm2(bottleneck_output))))
        )
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training
            )
        return new_features


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, quant):
        super(_Transition, self).__init__()
        self.add_module("norm", nn.BatchNorm2d(num_input_features))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module(
            "conv",
            nn.Conv2d(
                num_input_features,
                num_output_features,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )
        self.add_module("quant", quant())
        self.add_module("pool", nn.AvgPool2d(kernel_size=2, stride=2))
        self.add_module("quant", quant())


class _DenseBlock(nn.Module):
    def __init__(
        self,
        num_layers,
        num_input_features,
        bn_size,
        growth_rate,
        drop_rate,
        quant,
        efficient=False,
    ):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                quant=quant,
                efficient=efficient,
            )
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 3 or 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        small_inputs (bool) - set to True if images are 32x32. Otherwise assumes images are larger.
        efficient (bool) - set to True to use checkpointing. Much more memory efficient, but slower.
    """

    def __init__(
        self,
        growth_rate=12,
        block_config=(16, 16, 16),
        compression=0.5,
        num_init_features=24,
        bn_size=4,
        drop_rate=0,
        num_classes=10,
        small_inputs=True,
        efficient=False,
        forward_wl=-1,
        forward_fl=-1,
        backward_wl=-1,
        backward_fl=-1,
        forward_layer_type="fixed",
        backward_layer_type="",
        forward_round_type="stochastic",
        backward_round_type="",
    ):

        assert forward_layer_type in ["block", "fixed"]
        assert forward_round_type in ["nearest", "stochastic"]
        if backward_layer_type == "":
            backward_layer_type = forward_layer_type
        if backward_round_type == "":
            backward_round_type = forward_round_type
        assert backward_layer_type in ["block", "fixed"]
        assert backward_round_type in ["nearest", "stochastic"]

        if forward_layer_type == "block":
            quant = lambda: BlockQuantizer(
                forward_wl, backward_wl, forward_round_type, backward_round_type
            )
        elif forward_layer_type == "fixed":
            quant = lambda: FixedQuantizer(
                forward_wl,
                forward_fl,
                backward_wl,
                backward_fl,
                forward_round_type,
                backward_round_type,
            )

        super(DenseNet, self).__init__()
        assert 0 < compression <= 1, "compression of densenet should be between 0 and 1"
        self.avgpool_size = 8 if small_inputs else 7

        # First convolution
        if small_inputs:
            self.features = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv0",
                            nn.Conv2d(
                                3,
                                num_init_features,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False,
                            ),
                        ),
                    ]
                )
            )
        else:
            self.features = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv0",
                            nn.Conv2d(
                                3,
                                num_init_features,
                                kernel_size=7,
                                stride=2,
                                padding=3,
                                bias=False,
                            ),
                        ),
                    ]
                )
            )
            self.features.add_module("norm0", nn.BatchNorm2d(num_init_features))
            self.features.add_module("relu0", nn.ReLU(inplace=True))
            self.features.add_module(
                "pool0",
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False),
            )

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                quant=quant,
                efficient=efficient,
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=int(num_features * compression),
                    quant=quant,
                )
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = int(num_features * compression)

        # Final batch norm
        self.features.add_module("norm_final", nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Initialization
        for name, param in self.named_parameters():
            if "conv" in name and "weight" in name:
                n = param.size(0) * param.size(2) * param.size(3)
                param.data.normal_().mul_(math.sqrt(2.0 / n))
            elif "norm" in name and "weight" in name:
                param.data.fill_(1)
            elif "norm" in name and "bias" in name:
                param.data.fill_(0)
            elif "classifier" in name and "bias" in name:
                param.data.fill_(0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=self.avgpool_size).view(
            features.size(0), -1
        )
        out = self.classifier(out)
        return out


class DenseNetBC100LP:
    base = DenseNet
    args = list()
    kwargs = {
        "growth_rate": 12,
        "block_config": (16, 16, 16),
        "small_inputs": True,
        "efficient": False,
    }


class DenseNetBC190LP:
    base = DenseNet
    args = list()
    kwargs = {
        "growth_rate": 40,
        "block_config": (31, 31, 31),
        "small_inputs": True,
        "efficient": False,
    }
