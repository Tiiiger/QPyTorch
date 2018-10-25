import torch
import torch.nn as nn
from .quant_function import *
import numpy as np

__all__ = ['Quantizer']

class Quantizer(nn.Module):
    def __init__(self,
                 forward_number, backward_number,
                 forward_rounding, backward_rounding):
        super(Quantizer, self).__init__()
        self.quantize = quantizer(forward_number, backward_number,
                                 forward_rounding, backward_rounding)

    def forward(self, x):
        return self.quantize(x)
