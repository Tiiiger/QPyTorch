import torch
from torch.optim import Optimizer, SGD, Adam

__all__ = ["OptimLP"]

class OptimLP(Optimizer):
    """
    A low-precision optimizer wrapper that handles weight, gradient, accumulator quantization.

    Args:
        - :attr: `optim`: underlying optimizer to use
        - :attr: `weight_quant`: a weight quantization function which takes a pytorch tensor and returns a tensor. If None, does not quantize weight.
        - :attr: `grad_quant`: a gradient quantization function which takes a pytorch tensor and returns a tensor. If None, does not quantize weight.
        - :attr: `grad_scaling`: float, scaling factor before apply gradient quantization.
        - :attr: `momentum_quant`: a momentum quantization function which takes a pytorch tensor and returns a tensor.
                                   If None, does not quantize weight.
        - :attr: `acc_quant`: a accumulator quantization function which takes
                              a pytorch tensor and returns a tensor. If not None, a
                              OptimLP object would create memory copies of model parameters that serve as
                              gradient accumulators. If None, does not use gradient accumulators.

    Example:
        >>> weight_q = quantizer(...) # define weight quantization
        >>> optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer = OptimLP(optiimizer, weight_quant=weight_q)
    """

    def __init__(self, optim,
                 weight_quant=None,
                 grad_scaling=1.0,
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

        if isinstance(self.optim, SGD):
            self.momentum_keys = ['momentum_buffer']
        elif isinstance(self.optim, Adam):
            # TODO: support amsgrad
            self.momentum_keys = ['exp_avg', 'exp_avg_sq']
        else:
            raise NotImplementedError("Only supporting Adam and SGD for now. ")

        if self.acc_quant != None:
            self.weight_acc = {}
            for group in self.param_groups:
                for p in group['params']:
                    self.weight_acc[p] = p.detach().clone()

    def step(self, closure=None):
        """
        Performs one step of optimization with the underlying optimizer.
        Quantizes gradient and momentum before stepping. Quantizes gradient accumulator and weight after stepping.
        """
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
            for group in self.param_groups:
                if group['momentum'] == 0: continue
                for p in group['params']:
                    param_state = self.optim.state[p]
                    for key in self.momentum_keys:
                        param_state[key] = self.momentum_quant(param_state[key])

        return loss

    def __repr__(self):
        return "LP Optimizer: {}".format(self.optim.__repr__())

    def __str__(self):
        return "LP Optimizer: {}".format(self.optim.__str__())
