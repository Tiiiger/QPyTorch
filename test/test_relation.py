import torch
import unittest
from qtorch.quant import block_quantize, fixed_point_quantize, float_quantize

class TestRelation(unittest.TestCase):
    def test_fixed_block_zero_exponent(self):
        """
        invariant: when the max exponent of a block is zero, block floating point behaves similar to
        a fixed point where fl = wl -1, without the lowest number of the fixed point (-1). 
        """
        wl = 5
        fl = 4
        t_max = 1-(2**(-fl))
        to_quantize = torch.linspace(-t_max, t_max, steps=20, device='cuda')
        fixed_quantized = fixed_point_quantize(to_quantize, forward_wl=wl, forward_fl=fl, forward_rounding='nearest')
        block_quantized = block_quantize(to_quantize, forward_wl=wl, forward_rounding='nearest')
        self.assertTrue(torch.eq(fixed_quantized, block_quantized).all().item())
        wl = 3
        fl = 2
        t_max = 1-(2**(-fl))
        to_quantize = torch.linspace(-t_max, t_max, steps=100, device='cuda')
        fixed_quantized = fixed_point_quantize(to_quantize, forward_wl=wl, forward_fl=fl, forward_rounding='nearest')
        block_quantized = block_quantize(to_quantize, forward_wl=wl, forward_rounding='nearest')
        self.assertTrue(torch.eq(fixed_quantized, block_quantized).all())

    def test_block_float_same_exponent(self):
        """
        invariant: when there is only one kind of exponent in a block, block floating point behaves the same as
        a floating point
        """
        self.assertTrue(False)

    def test_float_half(self):
        """
        invariant: when quantizing into a half precision floating point, float quantize kernel behaves the same as the
        built-in half precision tensor
        """
        a = torch.rand(int(1e4)).cuda()
        man = 11
        exp = 5
        half_a = a.half()
        def compute_dist(a, half_a):
            diff =  (a-half_a.float())
            diff2 = diff**2
            return diff2.sum()
        sim_half_a = float_quantize(a, forward_man_bits=man, forward_exp_bits=exp, forward_rounding="nearest")
        self.assertEqual(compute_dist(a, sim_half_a), compute_dist(a, half_a))

if __name__ == "__main__":
    unittest.main()
