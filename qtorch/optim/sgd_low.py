import torch
from torch.optim import Optimizer, SGD

__all__ = ["SGDLP"]

class SGDLP(Optimizer):

    def __init__(self, sgd_optim,
                 grad_scaling=1,
                 weight_quant=None,
                 grad_quant=None,
                 momentum_quant=None):
        assert isinstance(sgd_optim, SGD)
        super(SGDLP, self).__init__(sgd_optim.param_groups, sgd_optim.defaults) # place holder

        # python dictionary does not copy by default
        self.param_groups = sgd_optim.param_groups
        self.sgd_optim = sgd_optim

        assert grad_scaling > 0, "gradient scaling must be positive"
        self.grad_scaling = 1

        self.weight_quant=weight_quant
        self.grad_quant=grad_quant
        self.momentum_quant=momentum_quant

    def step(self, closure=None):
        # quantize gradient
        for group in self.param_groups:
            for p in group['params']:
                if not self.grad_quant is None:
                    p.grad.data = self.grad_quant(p.grad.data*self.grad_scaling)

        loss = self.sgd_optim.step()

        # quantize weight
        if not self.weight_quant is None:
            for group in self.param_groups:
                for p in group['params']:
                    p.data = self.weight_quant(p.data)

        # quantize momentum
        if not self.momentum_quant is None:
            for group in self.param_groups:
                for p in group['params']:
                    param_state = self.sgd_optim.state[p]
                    param_state['momentum_buffer'] = self.momentum_quant(param_state['momentum_buffer'])

        return loss
