import torch
from torch.optim import Optimizer, SGD, Adam

__all__ = ["OptimLP"]

class OptimLP(Optimizer):

    def __init__(self, optim,
                 grad_scaling=1,
                 weight_quant=None,
                 grad_quant=None,
                 momentum_quant=None,
                 acc_quant=None):
        assert (isinstance(optim, SGD) or isinstance(optim, Adam))
        super(OptimLP, self).__init__(optim.param_groups, optim.defaults) # place holder

        # python dictionary does not copy by default
        self.param_groups = optim.param_groups
        self.optim = optim

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

        loss = self.optim.step()

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
            if isinstance(self.optim, SGD):
                keys = ['momentum_buffer']
            elif isinstance(self.optim, Adam):
                # TODO: support amsgrad
                keys = ['exp_avg', 'exp_avg_sq']
            for group in self.param_groups:
                if group['momentum'] == 0: continue
                for p in group['params']:
                    param_state = self.optim.state[p]
                    for key in keys:
                        param_state[key] = self.momentum_quant(param_state[key])

        return loss
