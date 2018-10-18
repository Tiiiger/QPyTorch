import torch
import quant_cuda
import quant_cpu
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__all__ = ['fixed_point_quantize', 'block_quantize', 'float_quantize', 'R', "quantize"]
R = torch.cuda.FloatTensor(int(1e8)).uniform_()

def assert_wl_fl(wl, fl, stage):
    if wl == -1 and fl != -1:
        raise ValueError("fixed point {} wl {}, fl {}".format(stage, wl, fl))

def make_r(x, random=R):
    size = x.numel()
    start = np.random.randint(0, random.size(0)-size-1)
    r = random[start:start+size]
    r = r.view_as(x)
    return r

class FixedPointRounding(torch.autograd.Function):

    @staticmethod
    def forward(self, x, forward_wl=-1, forward_fl=-1, backward_wl=-1, backward_fl=-1,
                forward_rounding="stochastic", backward_rounding="stochastic"):
        if x.is_cuda:
            quant_module = quant_cuda
        else:
            quant_module = quant_cpu

        self.backward_wl = backward_wl
        self.backward_fl = backward_fl
        self.backward_rounding = backward_rounding

        assert_wl_fl(forward_wl, forward_fl, "forward")
        assert_wl_fl(backward_wl, backward_fl, "backward")
        assert forward_rounding in ["stochastic", "nearest"]
        assert backward_rounding in ["stochastic", "nearest"]

        if forward_wl == -1: return x

        if forward_rounding=="nearest":
            out = quant_module.fixed_point_quantize_nearest(x, forward_wl, forward_fl)
        elif forward_rounding=="stochastic":
            r = make_r(x)
            out = quant_module.fixed_point_quantize_stochastic(x, r, forward_wl, forward_fl)
        return out

    @staticmethod
    def backward(self, grad_output):
        if grad_output.is_cuda:
            quant_module = quant_cuda
        else:
            quant_module = quant_cpu

        grad_input = None

        if self.needs_input_grad[0]:
            if self.backward_wl > 0:
                if self.backward_rounding=="nearest":
                    grad_input = quant_module.fixed_point_quantize_nearest(grad_output,
                                                                           self.backward_wl,
                                                                           self.backward_fl)
                elif self.backward_rounding=="stochastic":
                    r = make_r(x)
                    grad_input = quant_module.fixed_point_quantize_stochastic(grad_output, r,
                                                                              self.backward_wl,
                                                                              self.backward_fl)
            else:
                grad_input = grad_output

        return grad_input, None, None, None, None, None, None

class BlockRounding(torch.autograd.Function):
    @staticmethod
    def forward(self, x, forward_wl=-1, backward_wl=-1, forward_rounding="stochastic", backward_rounding="stochastic"):
        if x.is_cuda:
            quant_module = quant_cuda
        else:
            quant_module = quant_cpu

        self.backward_wl = backward_wl
        self.backward_rounding = backward_rounding

        assert forward_rounding in ["stochastic", "nearest"]
        assert backward_rounding in ["stochastic", "nearest"]

        if forward_wl == -1: return x
        if forward_rounding=="nearest":
            out = quant_module.block_quantize_nearest(x, forward_wl)
        elif forward_rounding=="stochastic":
            r = make_r(x)
            out = quant_module.block_quantize_stochastic(x, r, forward_wl)

        return out

    @staticmethod
    def backward(self, grad_output):
        if grad_output.is_cuda:
            quant_module = quant_cuda
        else:
            quant_module = quant_cpu

        if self.needs_input_grad[0]:
            if self.backward_wl > 0:
                if self.backward_rounding=="nearest":
                    grad_input = quant_module.block_quantize_nearest(grad_output, self.backward_wl)
                elif self.backward_rounding=="stochastic":
                    r = make_r(x)
                    grad_input = quant_module.block_quantize_stochastic(grad_output, r, self.backward_wl)
            else:
                grad_input = grad_output
        return grad_input, None, None, None, None

class FloatRounding(torch.autograd.Function):

    @staticmethod
    def forward(self, x, forward_man_bits=-1, forward_exp_bits=-1, backward_man_bits=-1, backward_exp_bits=-1,
                forward_rounding="stochastic", backward_rounding="stochastic", random=R):
        if x.is_cuda:
            quant_module = quant_cuda
        else:
            quant_module = quant_cpu
        self.backward_man_bits = backward_man_bits
        self.backward_exp_bits = backward_exp_bits
        self.backward_rounding = backward_rounding

        assert forward_rounding in ["stochastic", "nearest"]
        assert backward_rounding in ["stochastic", "nearest"]

        assert forward_man_bits <= 23
        assert backward_man_bits <= 23
        assert forward_exp_bits <= 8
        assert backward_exp_bits <= 8

        if forward_man_bits == -1: return x

        if forward_rounding=="nearest":
            out = quant_cuda.float_quantize_nearest(x, forward_man_bits, forward_exp_bits)
        elif forward_rounding=="stochastic":
            out = quant_cuda.float_quantize_stochastic(x, forward_man_bits, forward_exp_bits)
        return out

    @staticmethod
    def backward(self, grad_output):
        grad_input = None
        if grad_output.is_cuda:
            quant_module = quant_cuda
        else:
            quant_module = quant_cpu

        if self.needs_input_grad[0]:
            if self.backward_wl > 0:
                if self.backward_rounding=="nearest":
                    grad_input = quant_cuda.float_quantize_nearest(grad_output,
                                                                   self.backward_man_bits,
                                                                   self.backward_exp_bits)
                elif self.backward_rounding=="stochastic":
                    grad_input = quant_cuda.float_quantize_stochastic(grad_output,
                                                                      self.backward_man_bits,
                                                                      self.backward_exp_bits)
            else:
                grad_input = grad_output

        return grad_input, None, None, None, None, None, None

def quantize(forward_wl, forward_fl, backward_wl, backward_fl,
             forward_man, backward_man, forward_exp, backward_exp,
             forward_rounding, backward_rounding, forward_type, backward_type):
    for rounding in [forward_rounding, backward_rounding]:
        assert rounding in ["stochastic", "nearest"], "invalid rounding type".format(rounding)
    for num_type in [forward_type, backward_type]:
        assert num_type in ["fixed", "block", "float"], "invalid rounding type".format(rounding)

    assert forward_man <= 23
    assert backward_man <= 23
    assert forward_exp <= 8
    assert backward_exp <= 8

    class Rounding(torch.autograd.Function):
        @staticmethod
        def forward(self, x):
            if x.is_cuda:
                quant_module = quant_cuda
            else:
                quant_module = quant_cpu

            if forward_type=="block" or forward_type=="fixed":
                if forward_wl==-1: return x
            elif forward_type=="float":
                if forward_man==-1 and forward_exp==-1: return x

            if forward_rounding=="nearest":
                if forward_type=="block":
                    out = quant_module.block_quantize_nearest(x, forward_wl)
                elif forward_type=='fixed':
                    out = quant_module.fixed_point_quantize_nearest(x, forward_wl, forward_fl)
                elif forward_type=="float":
                    out = quant_module.float_quantize_nearest(x, forward_man, forward_exp)
            elif forward_rounding=="stochastic":
                if forward_type=="block":
                    r = make_r(x)
                    out = quant_module.block_quantize_stochastic(x, r, forward_wl)
                elif forward_type=="fixed":
                    r = make_r(x)
                    out = quant_module.fixed_point_quantize_stochastic(x, r, forward_wl, forward_fl)
                elif forward_type=="float":
                    out = quant_module.float_quantize_stochastic(x, forward_man, forward_exp)

            return out

        @staticmethod
        def backward(self, grad_output):
            if grad_output.is_cuda:
                quant_module = quant_cuda
            else:
                quant_module = quant_cpu

            if self.needs_input_grad[0]:
                if backward_wl > 0:
                    if backward_rounding=="nearest":
                        if backward_type=="block":
                            grad_input = quant_module.block_quantize_nearest(grad_output, backward_wl)
                        elif backward_type=="fixed":
                            grad_input = quant_module.fixed_point_quantize_nearest(grad_output, backward_wl, backward_fl)
                        elif backward_type=="float":
                            grad_input = quant_module.float_quantize_nearest(grad_output, backward_man, backward_exp)
                    elif backward_rounding=="stochastic":
                        if backward_type=="block":
                            r = make_r(grad_output)
                            grad_input = quant_module.block_quantize_stochastic(grad_output, r, backward_wl)
                        elif backward_type=="fixed":
                            r = make_r(grad_output)
                            grad_input = quant_module.fixed_point_quantize_stochastic(grad_output, r, backward_wl, backward_fl)
                        elif backward_type=="float":
                            grad_input = quant_module.float_quantize_stochastic(grad_output, backward_man, backward_exp)
                else:
                    grad_input = grad_output
            else:
                grad_input = None

            return grad_input, None, None, None, None

    return Rounding.apply

def fixed_point_quantize(x, forward_wl=-1, forward_fl=-1, backward_wl=-1, backward_fl=-1,
                         forward_rounding="stochastic", backward_rounding="stochastic"):
    return FixedPointRounding.apply(x, forward_wl, forward_fl, backward_wl, backward_fl,
                                    forward_rounding, backward_rounding)

def block_quantize(x, forward_wl=-1, backward_wl=-1, forward_rounding="stochastic",
                   backward_rounding="stochastic"):
    return BlockRounding.apply(x, forward_wl, backward_wl, forward_rounding,
                               backward_rounding)

def float_quantize(x, forward_man_bits=-1, forward_exp_bits=-1, backward_man_bits=-1,
                   backward_exp_bits=-1, forward_rounding="stochastic",
                   backward_rounding="stochastic"):
    return FloatRounding.apply(x, forward_man_bits, forward_exp_bits, backward_man_bits, backward_exp_bits,
                               forward_rounding, backward_rounding)
