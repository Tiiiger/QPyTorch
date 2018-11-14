import torch
import unittest
from qtorch.quant import *
from qtorch import FixedPoint, BlockFloatingPoint, FloatingPoint

class TestQuant(unittest.TestCase):
    """
    random test on the behavior of nearest fixed point rounding
    """
    def test_fixed_random(self):
        S = lambda bits :  2.**(bits-1)
        Q = lambda x, bits : torch.round(x*S(bits))/S(bits)
        wl = 8
        number = FixedPoint(wl=wl, fl=wl-1, clamp=False)
        quant = lambda x : fixed_point_quantize(x, number, "nearest")

        x = torch.randn(int(1e8), device='cuda')
        oracle = Q(x, wl)
        target = quant(x)
        mask = torch.eq(oracle, target)
        print((1-mask).sum())

if __name__ == "__main__":
    unittest.main()
