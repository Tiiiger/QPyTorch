import torch
import unittest
from qtorch.quant import *
from qtorch import FixedPoint, BlockFloatingPoint, FloatingPoint

class TestBackward(unittest.TestCase):
    """
    invariant: stochastic rounding is unbiased
    """
    def test_backward_fixed(self):
        for d in ['cpu', 'cuda']:
            number = FixedPoint(wl=5, fl=4)
            t_min = -2 ** (number.wl-number.fl-1)
            t_max = 2 ** (number.wl-number.fl-1) - 2 ** (-number.fl)
            a = torch.linspace(t_min, t_max, steps=100, device=d, requires_grad=True)
            quant = quantizer(forward_number=number, forward_rounding='nearest', backward_number=number, backward_rounding='nearest')
            s = quant(a).sum()
            s.backward()
            true_grad = torch.ones_like(a)*t_max
            self.assertTrue(torch.eq(a.grad, true_grad).all().item())

            a = torch.tensor([-1, -0.6, 0, 1], device=d, requires_grad=True)
            quant = quantizer(forward_number=number, forward_rounding='nearest', backward_number=number, backward_rounding='nearest', clamping_grad_zero=True)
            s = quant(a).sum()
            s.backward()
            true_grad = torch.ones_like(a)*t_max
            true_grad[-1] = 0
            self.assertTrue(torch.eq(a.grad, true_grad).all().item())

    def test_backward_block(self):
        for d in ['cpu', 'cuda']:
            number = BlockFloatingPoint(wl=5)
            a = torch.linspace(-0.9, 0.9, steps=100, device=d, requires_grad=True)
            quant = quantizer(forward_number=number, forward_rounding='nearest', backward_number=number, backward_rounding="nearest")
            s = quant(a).sum()
            s.backward()
            true_grad = torch.ones_like(a)
            self.assertTrue(torch.eq(a.grad, true_grad).all().item())

    def test_backward_float(self):
        for d in ['cpu', 'cuda']:
            number = FloatingPoint(exp=3, man=5)
            a = torch.linspace(-0.9, 0.9, steps=100, device=d, requires_grad=True)
            quant = quantizer(forward_number=number, forward_rounding='nearest', backward_number=number, backward_rounding="nearest")
            s = quant(a).sum()
            s.backward()
            true_grad = torch.ones_like(a)
            self.assertTrue(torch.eq(a.grad, true_grad).all().item())

if __name__ == "__main__":
    unittest.main()
