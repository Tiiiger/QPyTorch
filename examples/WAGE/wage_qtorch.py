import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
from torch.autograd import Function
from qtorch.quant import fixed_point_quantize, quantizer
from qtorch import FixedPoint

def shift(x):
    max_entry = x.abs().max()
    return x/2.**torch.ceil(torch.log2(max_entry))

def C(x, bits):
    if bits > 15 or bits == 1:
        delta = 0
    else:
        delta = 1. / (2.**(bits-1))
    upper = 1  - delta
    lower = -1 + delta
    return torch.clamp(x, lower, upper)

def QW(x, bits, scale=1.0, mode="nearest"):
    y = fixed_point_quantize(x, wl=bits, fl=bits-1, clamp=True, symmetric=True, rounding=mode)
    # per layer scaling
    if scale>1.8: y /= scale
    return y

def QG(x, bits_G, bits_R, lr, mode="nearest"):
    x = shift(x)
    lr = lr /(2.**(bits_G-1))
    norm = fixed_point_quantize(lr*x, wl=bits_G, fl=bits_G-1, clamp=False, symmetric=True, rounding=mode)
    return norm

class WAGEQuantizer(Module):
    def __init__(self, bits_A, bits_E,
                 A_mode="nearest", E_mode="nearest"):
        super(WAGEQuantizer, self).__init__()
        self.activate_number = FixedPoint(wl=bits_A, fl=bits_A-1, clamp=True, symmetric=True) if bits_A != -1 else None
        self.error_number = FixedPoint(wl=bits_E, fl=bits_E-1, clamp=True, symmetric=True) if bits_E != -1 else None
        self.quantizer = quantizer(forward_number=self.activate_number, forward_rounding=A_mode,
                                   backward_number=self.error_number, backward_rounding=E_mode,
                                   clamping_grad_zero=True, backward_hooks=[shift])

    def forward(self, x):
        return self.quantizer(x)
