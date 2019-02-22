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

    def test_clamp_zero(self):
        def S(bits):
            return 2.**(bits-1)

        def C(x, bits):
            if bits > 15 or bits == 1:
                delta = 0
            else:
                delta = 1. / S(bits)
                upper = 1  - delta
                lower = -1 + delta
                return torch.clamp(x, lower, upper)

        def Q(x, bits):
            assert bits != -1
            if bits==1:
                return torch.sign(x)
            if bits > 15:
                return x
            return torch.round(x*S(bits))/S(bits)

        class Rounding(torch.autograd.Function):
            @staticmethod
            def forward(self, x, bits_A, bits_E):
                self.bits_E = bits_E
                x = Q(x, bits_A)
                t_max = 1- 1./(2.**(bits_A-1))
                t_min = -1 + 1./(2.**(bits_A-1))
                mask = (x > t_max) + (x < t_min)
                x = torch.clamp(x, t_min, t_max)
                self.mask = mask

                return x

            @staticmethod
            def backward(self, grad_output):
                if self.needs_input_grad[0]:
                    grad_input = Q(C(grad_output, self.bits_E), self.bits_E).masked_fill_(self.mask, 0)
                return grad_input, None, None

        wl = 4
        number = FixedPoint(wl=wl, fl=wl-1, clamp=True, symmetric=True)
        oracle = lambda x : Rounding.apply(x, wl, wl)
        quant = quantizer(forward_number=number, forward_rounding='nearest',
                          backward_number=number, backward_rounding='nearest',
                          clamping_grad_zero=True)

        x = torch.linspace(-2, 2, steps=10, device='cpu', requires_grad=True)
        x.sum().backward()

        y = torch.linspace(-2, 2, steps=10, device='cpu', requires_grad=True)
        oracle(y).sum().backward()

        z = torch.linspace(-2, 2, steps=10, device='cpu', requires_grad=True)
        quant(z).sum().backward()

        self.assertTrue(torch.eq(y.grad, z.grad).all().item())

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
