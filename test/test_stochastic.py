import torch
import unittest
from qtorch.quant import *

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
        wl = 5
        fl = 4
        a = torch.linspace(- 2 ** (wl-fl-1), 2 ** (wl-fl-1) - 2 ** (-fl), steps=100, device='cuda')
        quant = lambda x : fixed_point_quantize(x, forward_wl=wl, forward_fl=fl, forward_rounding='stochastic')
        exp_a = self.calc_expectation(a, quant)
        self.assertTrue(((a-exp_a)**2).mean()<1e-8)

    def test_stochastic_block(self):
        wl = 5
        a = torch.linspace(-0.9, 0.9, steps=100, device='cuda')
        quant = lambda x : block_quantize(x, forward_wl=wl, forward_rounding='stochastic')
        exp_a = self.calc_expectation(a, quant)
        self.assertTrue(((a-exp_a)**2).mean() < 1e-8)

    def test_stochastic_float(self):
        exp = 3
        man = 5
        a = torch.linspace(-0.9, 0.9, steps=100, device='cuda')
        quant = lambda x : float_quantize(x, forward_man_bits=man, forward_exp_bits=exp, forward_rounding='stochastic')
        exp_a = self.calc_expectation(a, quant)
        self.assertTrue(((a-exp_a)**2).mean() < 1e-8)

if __name__ == "__main__":
    unittest.main()
