import torch
import unittest
from qtorch.quant import *
from qtorch import FixedPoint, BlockFloatingPoint, FloatingPoint

class TestStochastic(unittest.TestCase):
    """
    test when [clamp] is True, fixed point quantization will
    clamp the number; otherwise, it will not.
    """
    def test_stochastic_fixed(self):
        for d in ['cpu', 'cuda']:
            for r in ['stochastic', "nearest"]:
                number = FixedPoint(wl=5, fl=4, clamp=True, symmetric=False)
                t_min = - 2 ** (number.wl-number.fl-1)
                t_max = 2 ** (number.wl-number.fl-1) - 2 ** (-number.fl)
                a = torch.linspace(-2, 2, steps=100, device=d)
                quant = lambda x : fixed_point_quantize(x, number=number, rounding=r)
                clamp_a = quant(a)
                self.assertEqual(t_max, clamp_a.max().item())
                self.assertEqual(t_min, clamp_a.min().item())

                number = FixedPoint(wl=5, fl=4, clamp=False, symmetric=False)
                a = torch.linspace(-2, 2, steps=100, device=d)
                quant = lambda x : fixed_point_quantize(x, number=number, rounding=r)
                no_clamp_a = quant(a)
                self.assertLess(t_max, no_clamp_a.max().item())
                self.assertGreater(t_min, no_clamp_a.min().item())

if __name__ == "__main__":
    unittest.main()
