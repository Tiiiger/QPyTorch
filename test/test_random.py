import torch
import unittest
from qtorch.quant import *
from qtorch import FixedPoint, BlockFloatingPoint, FloatingPoint

class TestQuant(unittest.TestCase):
    """
    random test on the behavior of nearest fixed point rounding
    """
    def test_fixed_random(self):
        S = lambda bits :  2**(bits)
        Q = lambda x, bits : torch.round(x*S(bits))/S(bits)
        wl = 8
        quant = lambda x : fixed_point_quantize(x, wl=wl, fl=wl, clamp=False, rounding="nearest")

        N = int(1e8)
        x = torch.randn(N, device='cuda')
        oracle = Q(x, wl)
        target = quant(x)
        matched = torch.eq(oracle, target).all().item()
        self.assertTrue(matched)

if __name__ == "__main__":
    unittest.main()
