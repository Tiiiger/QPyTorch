import torch
import torch.nn as nn
from .quant_function import *
import numpy as np

__all__ = ['Quantizer']

class Quantizer(nn.Module):
    def __init__(self, forward_wl, forward_fl, backward_wl, backward_fl,
                 forward_man, backward_man, forward_exp, backward_exp,
                 forward_rounding, backward_rounding, forward_type, backward_type):
        super(Quantizer, self).__init__()
        self.quantize = quantize(forward_wl, forward_fl, backward_wl, backward_fl,
                                 forward_man, backward_man, forward_exp, backward_exp,
                                 forward_rounding, backward_rounding, forward_type, backward_type)

    def forward(self, x):
        return self.quantize(x)
