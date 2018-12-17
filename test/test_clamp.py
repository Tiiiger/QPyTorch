import torch
import unittest
from qtorch.quant import *
from qtorch import FixedPoint, BlockFloatingPoint, FloatingPoint

class TestStochastic(unittest.TestCase):
    """
    invariant: quantized numbers cannot be greater than the maximum representable number
    or lower than the maximum representable number
    """
    def test_fixed(self):
        """test fixed point clamping"""
        for d in ['cpu', 'cuda']:
            for r in ['stochastic', "nearest"]:
                wl = 5
                fl = 4
                t_min = - 2 ** (wl-fl-1)
                t_max = 2 ** (wl-fl-1) - 2 ** (-fl)
                a = torch.linspace(-2, 2, steps=100, device=d)
                clamp_a = fixed_point_quantize(a, wl=wl, fl=fl, clamp=True, rounding=r)
                self.assertEqual(t_max, clamp_a.max().item())
                self.assertEqual(t_min, clamp_a.min().item())

                a = torch.linspace(-2, 2, steps=100, device=d)
                no_clamp_a = fixed_point_quantize(a, wl=wl, fl=fl, clamp=False, rounding=r)
                self.assertLess(t_max, no_clamp_a.max().item())
                self.assertGreater(t_min, no_clamp_a.min().item())

    def test_float(self):
        """test floating point clamping"""
        formats=[(6, 9), (5, 10), (5, 2)]

        for exp, man in formats:
            for d in ['cpu', 'cuda']:
                for r in ['stochastic', "nearest"]:
                    # test positive
                    a_max = 2**(2**(exp-1))*(1-2**(-man-1))
                    a_min = 2**(-2**(exp-1)+1)
                    a = torch.Tensor([2**50, 2**(-50)]).to(device=d)
                    quant_a = float_quantize(a, exp=exp, man=man, rounding=r)
                    self.assertEqual(quant_a[0].item(), a_max)
                    self.assertAlmostEqual(quant_a[1].item(), a_min)

                    # test negative
                    a_max = -2**(2**(exp-1))*(1-2**(-man-1))
                    a_min = -2**(-2**(exp-1)+1)
                    a = torch.Tensor([-2**35, -2**(-35)]).to(device=d)
                    quant_a = float_quantize(a, exp=exp, man=man, rounding=r)
                    self.assertEqual(quant_a[0].item(), a_max)
                    self.assertAlmostEqual(quant_a[1].item(), a_min)

if __name__ == "__main__":
    unittest.main()
