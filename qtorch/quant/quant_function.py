import torch
import quant_cuda
import quant_cpu
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import number

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
    # def forward(self, x, forward_wl=-1, forward_fl=-1, backward_wl=-1, backward_fl=-1,
    #             forward_rounding="stochastic", backward_rounding="stochastic"):
    def forward(self, x, 
                forward_number=None, backward_number=None,
                forward_rounding="stochastic", backward_rounding="stochastic"):
        for num_type in [forward_number, backward_number]:
            assert type(num_type) in [number.FixedPoint, type(None)], "Number must be FixedPoint"
        
        if x.is_cuda:
            quant_module = quant_cuda
        else:
            quant_module = quant_cpu

        self.backward_number = backward_number
        self.backward_rounding = backward_rounding

        assert_wl_fl(forward_number.wl, forward_number.fl, "forward")
        assert_wl_fl(backward_number.wl, backward_number.fl, "backward")
        assert forward_rounding in ["stochastic", "nearest"]
        assert backward_rounding in ["stochastic", "nearest"]

        if type(forward_number) == type(None): return x

        if forward_rounding=="nearest":
            out = quant_module.fixed_point_quantize_nearest(x, forward_number.wl, forward_number.fl)
        elif forward_rounding=="stochastic":
            r = make_r(x).to(x.device)
            out = quant_module.fixed_point_quantize_stochastic(x, r, forward_number.wl, forward_number.fl)
        return out

    @staticmethod
    def backward(self, grad_output):
        if grad_output.is_cuda:
            quant_module = quant_cuda
        else:
            quant_module = quant_cpu

        grad_input = None

        if self.needs_input_grad[0]:
            if not self.backward_number == None:
                if self.backward_rounding=="nearest":
                    grad_input = quant_module.fixed_point_quantize_nearest(grad_output,
                                                                           self.backward_number.wl,
                                                                           self.backward_number.fl)
                elif self.backward_rounding=="stochastic":
                    r = make_r(x).to(x.device)
                    grad_input = quant_module.fixed_point_quantize_stochastic(grad_output, r,
                                                                              self.backward_number.wl,
                                                                              self.backward_number.fl)
            else:
                grad_input = grad_output

        return grad_input, None, None, None, None, None, None

class BlockRounding(torch.autograd.Function):
    @staticmethod
    def forward(self, x, 
                forward_number=None, backward_number=None, 
                forward_rounding="stochastic", backward_rounding="stochastic"):
        for num_type in [forward_number, backward_number]:
            assert type(num_type) in [number.BlockFloatingPoint, type(None)], "Number must be BlockFloatingPoint"

        if x.is_cuda:
            quant_module = quant_cuda
        else:
            quant_module = quant_cpu

        self.backward_number = self.backward_number
        self.backward_rounding = backward_rounding

        assert forward_rounding in ["stochastic", "nearest"]
        assert backward_rounding in ["stochastic", "nearest"]

        if forward_number == type(None): return x
        if forward_rounding=="nearest":
            out = quant_module.block_quantize_nearest(x, forward_number.wl)
        elif forward_rounding=="stochastic":
            r = make_r(x).to(x.device)
            out = quant_module.block_quantize_stochastic(x, r, forward_number.wl)

        return out

    @staticmethod
    def backward(self, grad_output):
        if grad_output.is_cuda:
            quant_module = quant_cuda
        else:
            quant_module = quant_cpu

        if self.needs_input_grad[0]:
            if not self.backward_number == None:
                if self.backward_rounding=="nearest":
                    grad_input = quant_module.block_quantize_nearest(grad_output, self.backward_number.wl)
                elif self.backward_rounding=="stochastic":
                    r = make_r(x).to(x.device)
                    grad_input = quant_module.block_quantize_stochastic(grad_output, r, self.backward_number.wl)
            else:
                grad_input = grad_output
        return grad_input, None, None, None, None

class FloatRounding(torch.autograd.Function):

    @staticmethod
    def forward(self, x, 
                forward_number=None, backward_number=None,
                forward_rounding="stochastic", backward_rounding="stochastic", 
                random=R):
        for num_type in [forward_number, backward_number]:
            assert type(num_type) in [number.FloatingPoint, type(None)], "Number must be FloatingPoint"
        
        if x.is_cuda:
            quant_module = quant_cuda
        else:
            quant_module = quant_cpu
        
        self.backward_number = backward_number
        self.backward_rounding = backward_rounding

        assert forward_rounding in ["stochastic", "nearest"]
        assert backward_rounding in ["stochastic", "nearest"]

        # assert forward_man_bits <= 23
        # assert backward_man_bits <= 23
        # assert forward_exp_bits <= 8
        # assert backward_exp_bits <= 8

        if forward_number == None: return x

        if forward_rounding=="nearest":
            out = quant_cuda.float_quantize_nearest(x, forward_number.man, forward_number.exp)
        elif forward_rounding=="stochastic":
            out = quant_cuda.float_quantize_stochastic(x, forward_number.man, forward_number.exp)
        return out

    @staticmethod
    def backward(self, grad_output):
        grad_input = None
        if grad_output.is_cuda:
            quant_module = quant_cuda
        else:
            quant_module = quant_cpu

        if self.needs_input_grad[0]:
            if self.backward_number == None:
                if self.backward_rounding=="nearest":
                    grad_input = quant_cuda.float_quantize_nearest(grad_output,
                                                                   self.backward_number.man,
                                                                   self.backward_number.exp)
                elif self.backward_rounding=="stochastic":
                    grad_input = quant_cuda.float_quantize_stochastic(grad_output,
                                                                      self.backward_number.man,
                                                                      self.backward_number.exp)
            else:
                grad_input = grad_output

        return grad_input, None, None, None, None, None, None

# def quantize(forward_wl, forward_fl, backward_wl, backward_fl,
#              forward_man, backward_man, forward_exp, backward_exp,
#              forward_rounding, backward_rounding, forward_type, backward_type):
def quantize(forward_number=None, backward_number=None,
             forward_rounding, backward_rounding):
    for rounding in [forward_rounding, backward_rounding]:
        assert rounding in ["stochastic", "nearest"], "invalid rounding type {:s}".format(rounding)
    # for num_type in [forward_type, backward_type]:
    #     assert num_type in ["fixed", "block", "float"], "invalid rounding type".format(rounding)
    for num_type in [forward_number, backward_number]:
        assert type(num_type) in [number.FixedPoint, 
                                  number.BlockFloatingPoint,
                                  number.FloatingPoint,
                                  type(None)], 
                                 "invalid number type {:s}".format(str(type(num_type)))


    class Rounding(torch.autograd.Function):
        @staticmethod
        def forward(self, x):
            if x.is_cuda:
                quant_module = quant_cuda
            else:
                quant_module = quant_cpu

            if type(forward_number)==type(None): return x

            if forward_rounding=="nearest":
                if type(forward_number)==number.BlockFloatingPoint:
                    out = quant_module.block_quantize_nearest(x, forward_number.wl)
                elif type(forward_number)==number.FixedPoint:
                    out = quant_module.fixed_point_quantize_nearest(x, forward_number.wl, forward_number.fl)
                elif type(forward_number)==number.FloatingPoint:
                    out = quant_module.float_quantize_nearest(x, forward_number.man, forward_number.exp)
            elif forward_rounding=="stochastic":
                if type(forward_number)==number.BlockFloatingPoint:
                    r = make_r(x)
                    out = quant_module.block_quantize_stochastic(x, r, forward_number.wl)
                elif type(forward_number)==number.FixedPoint:
                    r = make_r(x)
                    out = quant_module.fixed_point_quantize_stochastic(x, r, forward_number.wl, forward_number.fl)
                elif type(forward_number)==number.FloatingPoint:
                    out = quant_module.float_quantize_stochastic(x, forward_number.man, forward_number.exp)

            return out

        @staticmethod
        def backward(self, grad_output):
            if grad_output.is_cuda:
                quant_module = quant_cuda
            else:
                quant_module = quant_cpu

            if self.needs_input_grad[0]:
                if backward_number == None:
                    grad_input = grad_output
                else:
                    if backward_rounding=="nearest":
                        if type(backward_number)==number.BlockFloatingPoint:
                            grad_input = quant_module.block_quantize_nearest(grad_output, backward_number.wl)
                        elif type(backward_number)==number.FixedPoint:
                            grad_input = quant_module.fixed_point_quantize_nearest(grad_output, backward_number.wl, backward_number.fl)
                        elif type(backward_number)==number.FloatingPoint:
                            grad_input = quant_module.float_quantize_nearest(grad_output, backward_number.man, backward_number.exp)
                    elif backward_rounding=="stochastic":
                        if type(backward_number)==number.BlockFloatingPoint:
                            r = make_r(grad_output)
                            grad_input = quant_module.block_quantize_stochastic(grad_output, r, backward_number.wl)
                        elif type(backward_number)==number.FixedPoint:
                            r = make_r(grad_output)
                            grad_input = quant_module.fixed_point_quantize_stochastic(grad_output, r, backward_number.wl, backward_number.fl)
                        elif type(backward_number)==number.FloatingPoint:
                            grad_input = quant_module.float_quantize_stochastic(grad_output, backward_number.man, backward_number.exp)
            else:
                grad_input = None

            return grad_input, None, None, None, None

    return Rounding.apply

def fixed_point_quantize(x, 
                         forward_number=None, backward_number=None,
                         forward_rounding="stochastic", backward_rounding="stochastic"):
    return FixedPointRounding.apply(x, 
                                    forward_number, backward_number,
                                    forward_rounding, backward_rounding)

def block_quantize(x, 
                   forward_number=None, backward_number=None, 
                   forward_rounding="stochastic", backward_rounding="stochastic"):
    return BlockRounding.apply(x, 
                               forward_number, backward_number, 
                               forward_rounding, backward_rounding)

def float_quantize(x, 
                   forward_number=None, backward_number=None,
                   forward_rounding="stochastic", backward_rounding="stochastic"):
    return FloatRounding.apply(x, 
                               forward_number, backward_number, 
                               forward_rounding, backward_rounding)
