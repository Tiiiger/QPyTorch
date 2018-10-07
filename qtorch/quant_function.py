import torch
import quant_cuda
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

R = torch.cuda.FloatTensor(int(1e8)).uniform_()

def assert_wl_fl(wl, fl, stage):
    if wl == -1 and fl != -1:
        raise ValueError("fixed point {} wl {}, fl {}".format(stage, wl, fl))

class FixedPointRounding(torch.autograd.Function):

    @staticmethod
    def forward(self, x, forward_wl=-1, forward_fl=-1, backward_wl=-1, backward_fl=-1,
                forward_rounding="stochastic", backward_rounding="stochastic"):
        self.backward_wl = backward_wl
        self.backward_fl = backward_fl
        self.backward_rounding = backward_rounding

        assert_wl_fl(forward_wl, forward_fl, "forward")
        assert_wl_fl(backward_wl, backward_fl, "backward")
        assert forward_rounding in ["stochastic", "nearest"]
        assert backward_rounding in ["stochastic", "nearest"]

        if forward_wl == -1: return x

        if forward_rounding=="nearest":
            raise NotImplementedError("not implement nearest rounding.")
        elif forward_rounding=="stochastic":
            size = 1
            for n in x.size(): size *= n
            start = np.random.randint(0, R.size(0)-size-1)
            r = R[start:start+size].view_as(x)
            out = quant_cuda.fixed_point_quantize(x, r, forward_wl, forward_fl)
        return out

    @staticmethod
    def backward(self, grad_output):
        grad_input = None

        if self.needs_input_grad[0]:
            if self.backward_wl > 0:
                if self.backward_rounding=="nearest":
                    raise NotImplementedError("not implement nearest rounding.")
                elif self.backward_rounding=="stochastic":
                    size = 1
                    for n in grad_output.size(): size *= n
                    start = np.random.randint(0, R.size(0)-size-1)
                    r = R[start:start+size].view_as(grad_output)

                    grad_input = quant_cuda.fixed_point_quantize(grad_output, r,
                                                                 self.backward_wl,
                                                                 self.backward_fl)
            else:
                grad_input = grad_output

        return grad_input, None, None, None, None, None, None

class BlockRounding(torch.autograd.Function):
    @staticmethod
    def forward(self, x, forward_wl=-1, backward_wl=-1, forward_rounding="stochastic", backward_rounding="stochastic"):
        self.backward_wl = backward_wl
        self.backward_rounding = backward_rounding

        assert forward_rounding in ["stochastic", "nearest"]
        assert backward_rounding in ["stochastic", "nearest"]

        if forward_wl == -1: return x
        if forward_rounding=="nearest":
            raise NotImplementedError("not implement nearest rounding.")
        elif forward_rounding=="stochastic":
            size = 1
            for n in x.size(): size *= n
            start = np.random.randint(0, R.size(0)-size-1)
            r = R[start:start+size].view_as(x)
            out = quant_cuda.block_quantize(x, r, forward_wl)

        return out

    @staticmethod
    def backward(self, grad_output):
        if self.needs_input_grad[0]:
            if self.backward_wl > 0:
                if self.backward_rounding=="nearest":
                    raise NotImplementedError("not implement nearest rounding.")
                elif self.backward_rounding=="stochastic":
                    size = 1
                    for n in grad_output.size(): size *= n
                    start = np.random.randint(0, R.size(0)-size-1)
                    r = R[start:start+size].view_as(grad_output)
                    grad_input = quant_cuda.block_quantize(grad_output, r, self.backward_wl)
            else:
                grad_input = grad_output
        return grad_input, None, None, None, None

def fixed_point_quantize(x, forward_wl=-1, forward_fl=-1, backward_wl=-1, backward_fl=-1,
                         forward_rounding="stochastic", backward_rounding="stochastic"):
    return FixedPointRounding.apply(x, forward_wl, forward_fl, backward_wl, backward_fl,
                forward_rounding, backward_rounding)

def block_quantize(x, forward_wl=-1, backward_wl=-1, forward_rounding="stochastic",
                   backward_rounding="stochastic"):
    return BlockRounding.apply(x, forward_wl, backward_wl, forward_rounding,
                backward_rounding)
