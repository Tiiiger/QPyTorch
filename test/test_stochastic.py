import torch
import unittest
from qtorch.quant import *
from qtorch import FixedPoint, BlockFloatingPoint, FloatingPoint

class TestStochastic(unittest.TestCase):
    """
    invariant: stochastic rounding is unbiased
    """
    def calc_expectation_error(self, a, quant, N):
        b = torch.zeros_like(a)
        for i in range(int(N)):
            b = b * i/(i+1.) + quant(a) / (i+1)
        error = ((a-b)**2).mean().cpu().item()
        return error

    def test_stochastic_fixed(self):
        for wl, fl in [(7,6)]:
            for d in ['cpu', 'cuda']:
                a = torch.linspace(-0.9, 0.9, steps=100, device=d)
                quant = lambda x : fixed_point_quantize(x, wl=wl, fl=fl, clamp=True, symmetric=False)
                error = self.calc_expectation_error(a, quant, 1e5)
                self.assertTrue(error<1e-6)
                number = FixedPoint(wl=wl, fl=fl, clamp=True, symmetric=False)
                quant = quantizer(forward_number=number, forward_rounding="stochastic")
                error = self.calc_expectation_error(a, quant, 1e5)
                self.assertTrue(error<1e-6)

    def test_stochastic_block(self):
        for d in ['cpu', 'cuda']:
            for dim in [-1, 0, 1]:
                a = torch.randn(10, 10, 10)
                quant = lambda x : block_quantize(x, wl=5, dim=dim)
                error = self.calc_expectation_error(a, quant, 1e5)
                self.assertTrue((error < 1e-6))
                number = BlockFloatingPoint(wl=5, dim=dim)
                quant = quantizer(forward_number=number, forward_rounding="stochastic")
                error = self.calc_expectation_error(a, quant, 1e5)
                self.assertTrue(error<1e-6)

    def test_stochastic_float(self):
        for d in ['cpu', 'cuda']:
            a = torch.rand(100).to(device=d)
            quant = lambda x : float_quantize(x, exp=6, man=5)
            error = self.calc_expectation_error(a, quant, 1e5)
            self.assertTrue((error < 1e-6))
            number = FloatingPoint(exp=6, man=5)
            quant = quantizer(forward_number=number, forward_rounding="stochastic")
            error = self.calc_expectation_error(a, quant, 1e5)
            self.assertTrue(error<1e-6)

if __name__ == "__main__":
    unittest.main()
