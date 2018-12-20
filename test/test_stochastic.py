import torch
import unittest
from qtorch.quant import *
from qtorch import FixedPoint, BlockFloatingPoint, FloatingPoint

class TestStochastic(unittest.TestCase):
    """
    invariant: stochastic rounding is unbiased
    """
    def calc_expectation(self, a, quant):
        b = torch.zeros_like(a)
        num_of_trails = int(1e5)
        for i in range(num_of_trails):
            b = b * i/(i+1.) + quant(a) / (i+1)
        return b

    def test_stochastic_fixed(self):
        for d in ['cpu', 'cuda']:
            number = FixedPoint(wl=5, fl=4)
            a = torch.linspace(- 2 ** (number.wl-number.fl-1), 2 ** (number.wl-number.fl-1) - 2 ** (-number.fl), steps=100, device=d)
            quant = quantizer(forward_number=number, forward_rounding='stochastic')
            exp_a = self.calc_expectation(a, quant)
            self.assertTrue(((a-exp_a)**2).mean()<1e-8)

            number = FixedPoint(wl=5, fl=4, clamp=True)
            a = torch.linspace(- 2 ** (number.wl-number.fl-1), 2 ** (number.wl-number.fl-1) - 2 ** (-number.fl), steps=100, device=d)
            quant = quantizer(forward_number=number, forward_rounding='stochastic', clamping_grad_zero=True)
            exp_a = self.calc_expectation(a, quant)
            self.assertTrue(((a-exp_a)**2).mean().item() <1e-8)

    def test_stochastic_block(self):
        for d in ['cpu', 'cuda']:
            number = BlockFloatingPoint(wl=6)
            a = torch.linspace(-0.9, 0.9, steps=100, device='cuda')
            quant = quantizer(forward_number=number, forward_rounding='stochastic')
            exp_a = self.calc_expectation(a, quant)
            diff = ((a-exp_a)**2).mean().item()
            self.assertTrue((diff < 1e-8))

    def test_stochastic_float(self):
        for d in ['cpu', 'cuda']:
            number = FloatingPoint(exp=5, man=3)
            a = torch.linspace(-0.9, 0.9, steps=100, device='cuda')
            quant = quantizer(forward_number=number, forward_rounding='stochastic')
            exp_a = self.calc_expectation(a, quant)
            diff = ((a-exp_a)**2).mean().item()
            self.assertTrue(diff < 1e-8)

if __name__ == "__main__":
    unittest.main()
