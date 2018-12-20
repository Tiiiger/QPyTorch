import torch
from qtorch import Number, FixedPoint, BlockFloatingPoint, FloatingPoint
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.cpp_extension import load
import os
current_path = os.path.dirname(os.path.realpath(__file__))
quant_cpu = load(
        name='quant_cpu',
        sources=[
            os.path.join(current_path, "quant_cpu/quant_cpu.cpp"),
            os.path.join(current_path, "quant_cpu/bit_helper.cpp"),
            os.path.join(current_path, "quant_cpu/sim_helper.cpp"),
            ]
        )
quant_cuda = load(
        name='quant_cuda',
        sources=[
            os.path.join(current_path, "quant_cuda/quant_cuda.cpp"),
            os.path.join(current_path, "quant_cuda/bit_helper.cu"),
            os.path.join(current_path, "quant_cuda/sim_helper.cu"),
            os.path.join(current_path, "quant_cuda/block_kernel.cu"),
            os.path.join(current_path, "quant_cuda/float_kernel.cu"),
            os.path.join(current_path, "quant_cuda/fixed_point_kernel.cu"),
            os.path.join(current_path, "quant_cuda/quant.cu"),
            ]
        )

__all__ = ['fixed_point_quantize', 'block_quantize', 'float_quantize', "quantizer"]

def assert_wl_fl(wl, fl, stage=""):
    if wl == -1 and fl != -1:
        raise ValueError("fixed point {} wl {}, fl {}".format(stage, wl, fl))

def get_module(x):
    if x.is_cuda:
        quant_module = quant_cuda
    else:
        quant_module = quant_cpu
    return quant_module

def quantizer(forward_number=None, backward_number=None,
              forward_rounding="stochastic", backward_rounding="stochastic",
              clamping_grad_zero=False, backward_hooks=[]):

    for rounding in [forward_rounding, backward_rounding]:
        assert rounding in ["stochastic", "nearest"], "invalid rounding type {:s}".format(rounding)
    for num in [forward_number, backward_number]:
        if num != None: assert isinstance(num, Number)

    if clamping_grad_zero==False:
        if forward_rounding=="nearest":
            if type(forward_number)==BlockFloatingPoint:
                forward_quant = lambda x, quant_module: quant_module.block_quantize_nearest(x, forward_number.wl)
            elif type(forward_number)==FixedPoint:
                forward_quant = lambda x, quant_module: quant_module.fixed_point_quantize_nearest(x, forward_number.wl,
                                                                                                  forward_number.fl, forward_number.clamp,
                                                                                                  forward_number.symmetric)
            elif type(forward_number)==FloatingPoint:
                forward_quant = lambda x, quant_module: quant_module.float_quantize_nearest(x, forward_number.man, forward_number.exp)
        elif forward_rounding=="stochastic":
            if type(forward_number)==BlockFloatingPoint:
                forward_quant = lambda x, quant_module: quant_module.block_quantize_stochastic(x, forward_number.wl)
            elif type(forward_number)==FixedPoint:
                forward_quant = lambda x, quant_module: quant_module.fixed_point_quantize_stochastic(x, forward_number.wl, forward_number.fl,
                                                                                                     forward_number.clamp, forward_number.symmetric)
            elif type(forward_number)==FloatingPoint:
                forward_quant = lambda x, quant_module: quant_module.float_quantize_stochastic(x, forward_number.man, forward_number.exp)
    else:
        if type(forward_number)==FixedPoint or forward_number==None:
            assert forward_number==None or forward_number.clamp == True, "must use clamping if zeroing out clamped gradient"
            if forward_rounding=="nearest":
                forward_quant = lambda x, quant_module: quant_module.fixed_point_quantize_nearest_mask(x, forward_number.wl, forward_number.fl, forward_number.symmetric)
            elif forward_rounding=="stochastic":
                forward_quant = lambda x, quant_module: quant_module.fixed_point_quantize_stochastic_mask(x, forward_number.wl, forward_number.fl, forward_number.symmetric)
        else:
            raise ValueError("zeroing clamping gradient only support fixed point.")

    if backward_rounding=="nearest":
        if type(backward_number)==BlockFloatingPoint:
            backward_quant = lambda a, quant_module: quant_module.block_quantize_nearest(a, backward_number.wl)
        elif type(backward_number)==FixedPoint:
            backward_quant = lambda a, quant_module: quant_module.fixed_point_quantize_nearest(a, backward_number.wl, backward_number.fl,
                                                                                               backward_number.clamp, backward_number.symmetric)
        elif type(backward_number)==FloatingPoint:
            backward_quant = lambda a, quant_module: quant_module.float_quantize_nearest(a, backward_number.man, backward_number.exp)
    elif backward_rounding=="stochastic":
        if type(backward_number)==BlockFloatingPoint:
            backward_quant = lambda a, quant_module: quant_module.block_quantize_stochastic(a, backward_number.wl)
        elif type(backward_number)==FixedPoint:
            backward_quant = lambda a, quant_module: quant_module.fixed_point_quantize_stochastic(a, backward_number.wl, backward_number.fl,
                                                                                                  backward_number.clamp, backward_number.symmetric)
        elif type(backward_number)==FloatingPoint:
            backward_quant = lambda a, quant_module: quant_module.float_quantize_stochastic(a, backward_number.man, backward_number.exp)

    if clamping_grad_zero == False:
        class Rounding(torch.autograd.Function):
            @staticmethod
            def forward(self, x):
                if forward_number==None: return x

                quant_module = get_module(x)
                out = forward_quant(x.contiguous(), quant_module)

                return out

            @staticmethod
            def backward(self, grad_output):
                if self.needs_input_grad[0]:
                    if backward_number == None:
                        grad_input = grad_output
                    else:
                        quant_module = get_module(grad_output)
                        grad_input = backward_quant(grad_output.contiguous(), quant_module)
                else:
                    grad_input = None

                return grad_input
    else:
        class Rounding(torch.autograd.Function):
            @staticmethod
            def forward(self, x):
                if forward_number==None:
                    self.mask = torch.zeros_like(x).byte()
                    return x
                else:
                    quant_module = get_module(x)
                    out, mask = forward_quant(x.contiguous(), quant_module)
                    self.mask = mask

                return out

            @staticmethod
            def backward(self, grad_output):
                if self.needs_input_grad[0]:
                    if backward_number == None:
                        grad_input = grad_output
                    else:
                        quant_module = get_module(grad_output)
                        # grad_output = grad_output.contiguous().masked_fill_(self.mask, 0)
                        for f in backward_hooks:
                            grad_output = f(grad_output)
                        grad_input = backward_quant(grad_output.contiguous(), quant_module).masked_fill(self.mask, 0)
                else:
                    grad_input = None

                return grad_input

    return Rounding.apply


def fixed_point_quantize(x, wl, fl, clamp=True, symmetric=False, rounding="stochastic"):
    assert isinstance(x, torch.Tensor)
    assert rounding in ["stochastic", "nearest"]
    assert_wl_fl(wl, fl)
    quant_module = get_module(x)
    if rounding == "nearest":
        out = quant_module.fixed_point_quantize_nearest(x.contiguous(), wl, fl, clamp, symmetric)
    elif rounding == "stochastic":
        out = quant_module.fixed_point_quantize_stochastic(x.contiguous(), wl, fl, clamp, symmetric)
    return out

def block_quantize(x, wl, rounding="stochastic"):
    assert isinstance(x, torch.Tensor), "x is not a Floating Point Tensor"
    assert rounding in ["stochastic", "nearest"], "invalid rounding mode, {}".format(rounding)
    quant_module = get_module(x)
    if rounding=="nearest":
        out = quant_module.block_quantize_nearest(x.contiguous(), wl)
    elif rounding=="stochastic":
        out = quant_module.block_quantize_stochastic(x.contiguous(), wl)
    return out

def float_quantize(x, exp, man, rounding="stochastic"):
    """Quantize a single precision Floating Point into low-precision Floating Point
    Args:
        x: the single precision number(torch.Tensor) to be quantized
        exp: number of bits allocated for exponent
        man: number of bits allocated for mantissa, not counting the virtual bit
        rounding: rounding mode, \"stochastic\" or \"nearest\"
    Return:
        a quantized low-precision floating point number as torch.Tensor
    """
    assert isinstance(x, torch.Tensor), "x is not a Floating Point Tensor"
    assert rounding in ["stochastic", "nearest"], "invalid rounding mode, {}".format(rounding)
    quant_module = get_module(x)
    if rounding=="nearest":
        out = quant_module.float_quantize_nearest(x.contiguous(), man, exp)
    elif rounding=="stochastic":
        out = quant_module.float_quantize_stochastic(x.contiguous(), man, exp)
    return out
