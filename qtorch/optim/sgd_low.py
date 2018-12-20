import torch
from torch.optim import Optimizer, SGD

__all__ = ["SGDLP"]

class SGDLP(Optimizer):

    def __init__(self, sgd_optim,
                 grad_scaling=1,
                 weight_quant=None,
                 grad_quant=None,
                 momentum_quant=None,
                 acc_quant=None):
        assert isinstance(sgd_optim, SGD)
        super(SGDLP, self).__init__(sgd_optim.param_groups, sgd_optim.defaults) # place holder

        # python dictionary does not copy by default
        self.param_groups = sgd_optim.param_groups
        self.sgd_optim = sgd_optim

        assert grad_scaling > 0, "gradient scaling must be positive"
        self.grad_scaling = grad_scaling

        self.weight_quant=weight_quant
        self.grad_quant=grad_quant
        self.momentum_quant=momentum_quant
        self.acc_quant=acc_quant

        if self.acc_quant != None:
            self.weight_acc = {}
            for group in self.param_groups:
                for p in group['params']:
                    self.weight_acc[p] = p.detach().clone()

    def step(self, closure=None):
        # quantize gradient
        if not self.grad_quant is None:
            for group in self.param_groups:
                for p in group['params']:
                    p.grad.data = self.grad_quant(p.grad.data*self.grad_scaling)

        # switch acc into weight before stepping
        if not self.acc_quant is None:
            for group in self.param_groups:
                for p in group['params']:
                    p.data = self.weight_acc[p].data

        loss = self.sgd_optim.step()

        # switch weight into acc after stepping and quantize
        if not self.acc_quant is None:
            for group in self.param_groups:
                for p in group['params']:
                    p.data = self.weight_acc[p].data = self.acc_quant(p.data).data

        # quantize weight from acc
        if not self.weight_quant is None:
            for group in self.param_groups:
                for p in group['params']:
                    p.data = self.weight_quant(p.data).data

        # quantize momentum
        if not self.momentum_quant is None:
            for group in self.param_groups:
                for p in group['params']:
                    param_state = self.sgd_optim.state[p]
                    param_state['momentum_buffer'] = self.momentum_quant(param_state['momentum_buffer'])

        return loss
