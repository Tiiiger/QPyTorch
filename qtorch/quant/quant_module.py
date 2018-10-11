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
        for rounding in [forward_rounding, backward_rounding]:
            assert rounding in ["stochastic", "nearest"], "invalid rounding type".format(rounding)
        for num_type in [forward_type, backward_type]:
            assert num_type in ["fixed", "block", "float"], "invalid rounding type".format(rounding)

        class Rounding(torch.autograd.Function):
            @staticmethod
            def forward(self, x):

                if forward_type=="block" or forward_type=="fixed":
                    if forward_wl==-1: return x
                elif forward_type=="float":
                    if forward_man==-1 and forward_exp==-1: return x

                if forward_rounding=="nearest":
                    raise NotImplementedError("not implement nearest rounding.")
                elif forward_rounding=="stochastic":
                    size = 1
                    for n in x.size(): size *= n
                    start = np.random.randint(0, R.size(0)-size-1)
                    r = R[start:start+size].view_as(x)
                    if forward_type=="block":
                        out = quant_cuda.block_quantize(x, r, forward_wl)
                    elif forward_type=="fixed":
                        out = quant_cuda.fixed_point_quantize(x, r, forward_wl, forward_fl)
                    elif forward_type=="float":
                        out = quant_cuda.float_point_quantize_stochastic(x, r, forward_wl, forward_fl)
                    return out

            @staticmethod
            def backward(self, grad_output):
                if self.needs_input_grad[0] and backward_wl > 0:
                    if backward_rounding=="nearest":
                        raise NotImplementedError("not implement nearest rounding.")
                    elif backward_rounding=="stochastic":
                        size = 1
                        for n in grad_output.size(): size *= n
                        start = np.random.randint(0, R.size(0)-size-1)
                        r = R[start:start+size].view_as(grad_output)
                        if backward_type=="block":
                            grad_input = quant_cuda.block_quantize(grad_output, r, backward_wl)
                        elif backward_type=="fixed":
                            grad_input = quant_cuda.fixed_point_quantize(grad_output, r, backward_wl, backward_fl)
                else:
                    grad_input = grad_output
                return grad_input, None, None, None, None

        self.quantize = Rounding.apply

    def forward(self, x):
        return self.quantize(x)
