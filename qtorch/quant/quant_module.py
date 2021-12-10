import torch
import torch.nn as nn
from qtorch import BlockFloatingPoint, FloatingPoint
from .quant_function import *
import numpy as np

__all__ = ["Quantizer"]


class Quantizer(nn.Module):
    def __init__(
        self,
        forward_number=None,
        backward_number=None,
        forward_rounding="stochastic",
        backward_rounding="stochastic",
        dynamic_precision=False,
    ):
        if dynamic_precision == True:
            if type(forward_number) == FloatingPoint or type(forward_number) == BlockFloatingPoint:
                print("Not support DYNAMIC PRECISION in Floating point")

        super(Quantizer, self).__init__()
        self.quantize = quantizer(
            forward_number, backward_number, forward_rounding, backward_rounding, dynamic_precision,
        )

    def forward(self, x):
        return self.quantize(x)
