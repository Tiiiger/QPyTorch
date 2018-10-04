import torch
import torch.nn as nn
from .quant_function import *

class BlockQuantizer(nn.Module):
    def __init__(self, forward_wl, backward_wl, forward_mode, backward_mode):
        super(BlockQuantizer, self).__init__()
        self.forward_wl=forward_wl
        self.backward_wl=backward_wl
        self.forward_mode=forward_mode
        self.backward_mode=backward_mode

    def forward(self, x):
        return block_quantize(x, self.forward_wl, self.backward_wl,
                              self.forward_mode, self.backward_mode)

class FixedQuantizer(nn.Module):
    def __init__(self, forward_wl, forward_fl, backward_wl, backward_fl, forward_mode, backward_mode):
        super(FixedQuantizer, self).__init__()
        self.forward_wl=forward_wl
        self.forward_fl=forward_fl
        self.backward_wl=backward_wl
        self.backward_fl=backward_fl
        self.forward_mode=forward_mode
        self.backward_mode=backward_mode

    def forward(self, x):
        return fixed_point_quantize(x, self.forward_wl, self.forward_fl,
                              self.backward_wl, self.backward_fl,
                              self.forward_mode, self.backward_mode)
