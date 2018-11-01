import torch
import unittest
import qtorch
import pdb; pdb.set_trace()  # breakpoint 37615112 //

from qtorch.quant import block_quantize, fixed_point_quantize, float_quantize
from qtorch import FixedPoint, BlockFloatingPoint, FloatingPoint

class TestDevice(unittest.TestCase):
    """
        invariant: cuda and cpp implementation should behave the same
    """
    def test_fixed_point(self):
        wl = 5
        fl = 4
        t_max = 1-(2**(-fl))
        to_quantize_cuda = torch.linspace(-t_max, t_max, steps=20, device='cuda')
        to_quantize_cpu = to_quantize_cuda.clone().to("cpu")
        fixed_quantized_cuda = fixed_point_quantize(to_quantize_cuda, forward_number=FixedPoint(wl=wl, fl=fl), forward_rounding='nearest')
        fixed_quantized_cpu = fixed_point_quantize(to_quantize_cpu, forward_number=FixedPoint(wl=wl, fl=fl), forward_rounding='nearest')
        self.assertTrue(torch.eq(fixed_quantized_cuda, fixed_quantized_cpu).all().item())
        wl = 3
        fl = 2
        t_max = 1-(2**(-fl))
        to_quantize_cuda = torch.linspace(-t_max, t_max, steps=100, device='cuda')
        to_quantize_cpu = to_quantize_cuda.clone().to("cpu")
        fixed_quantized_cuda = fixed_point_quantize(to_quantize_cuda, forward_number=FixedPoint(wl=wl, fl=fl), forward_rounding='nearest')
        fixed_quantized_cpu = fixed_point_quantize(to_quantize_cpu, forward_number=FixedPoint(wl=wl, fl=fl), forward_rounding='nearest')
        self.assertTrue(torch.eq(fixed_quantized_cuda, fixed_quantized_cpu).all())

    def test_fixed_point_stochastic(self):
        wl = 5
        fl = 4
        t_max = 1-(2**(-fl))
        t_min = (2**(-fl))
        to_quantize_cuda = torch.rand(1).cuda()
        to_quantize_cpu = to_quantize_cuda.clone().to("cpu")
        fixed_quantized_cuda = fixed_point_quantize(to_quantize_cuda, forward_number=FixedPoint(wl=wl, fl=fl), forward_rounding='stochastic')
        fixed_quantized_cpu = fixed_point_quantize(to_quantize_cpu, forward_number=FixedPoint(wl=wl, fl=fl), forward_rounding='stochastic')
        self.assertTrue(torch.abs(fixed_quantized_cuda - fixed_quantized_cpu).item() < t_min)
        
        wl = 3
        fl = 2
        t_max = 1-(2**(-fl))
        t_min = (2**(-fl))
        to_quantize_cuda = torch.rand(1).cuda()
        to_quantize_cpu = to_quantize_cuda.clone().to("cpu")
        fixed_quantized_cuda = fixed_point_quantize(to_quantize_cuda, forward_number=FixedPoint(wl=wl, fl=fl), forward_rounding='stochastic')
        fixed_quantized_cpu = fixed_point_quantize(to_quantize_cpu, forward_number=FixedPoint(wl=wl, fl=fl), forward_rounding='stochastic')
        self.assertTrue(torch.abs(fixed_quantized_cuda - fixed_quantized_cpu).item() < t_min)

    def test_block_floating_point_stochastic(self):
        self.assertTrue(False)

    def test_block_floating_point(self):
        """
        invariant: when there is only one kind of exponent in a block, block floating point behaves the same as
        a floating point
        """
        wl = 5
        t_max = 1-(2**(-4))
        to_quantize_cuda = torch.linspace(-t_max, t_max, steps=20, device='cuda')
        to_quantize_cpu = to_quantize_cuda.clone().to("cpu")
        block_quantized_cuda = block_quantize(to_quantize_cuda, forward_number=BlockFloatingPoint(wl=wl), forward_rounding='nearest')
        block_quantized_cpu = block_quantize(to_quantize_cpu, forward_number=BlockFloatingPoint(wl=wl), forward_rounding='nearest')
        self.assertTrue(torch.eq(block_quantized_cuda, block_quantized_cpu).all())

        wl = 3
        t_max = 1-(2**(-2))
        to_quantize_cuda = torch.linspace(-t_max, t_max, steps=100, device='cuda')
        to_quantize_cpu = to_quantize_cuda.clone().to("cpu")
        block_quantized_cuda = block_quantize(to_quantize_cuda, forward_number=BlockFloatingPoint(wl=wl), forward_rounding='nearest')
        block_quantized_cpu = block_quantize(to_quantize_cpu, forward_number=BlockFloatingPoint(wl=wl), forward_rounding='nearest')
        self.assertTrue(torch.eq(block_quantized_cuda, block_quantized_cpu).all())
        # self.assertTrue(False), "bad"

    def test_float_half(self):
        """
        invariant: when quantizing into a half precision floating point, float quantize kernel behaves the same as the
        built-in half precision tensor
        """
        a_cuda = torch.rand(int(1e4)).cuda()
        a_cpu = a_cuda.clone().to("cpu")
        man = 11
        exp = 5
        def compute_dist(a, half_a):
            diff =  (a-half_a.float())
            diff2 = diff**2
            return diff2.sum()
        sim_half_a_cuda = float_quantize(a_cuda, forward_number=FloatingPoint(exp=exp, man=man), forward_rounding="nearest")
        sim_half_a_cpu = float_quantize(a_cuda, forward_number=FloatingPoint(exp=exp, man=man), forward_rounding="nearest")
        self.assertEqual(compute_dist(a, sim_half_a_cuda), compute_dist(a, sim_half_a_cpu))

        a_cuda = torch.rand(int(1e4)).cuda()
        a_cpu = a_cuda.clone().to("cpu")
        man = 15
        exp = 4
        q_a_cuda = float_quantize(a_cuda, forward_number=FloatingPoint(exp=exp, man=man), forward_rounding="nearest")
        q_a_cpu = float_quantize(a_cuda, forward_number=FloatingPoint(exp=exp, man=man), forward_rounding="nearest")
        self.assertEqual(compute_dist(a, q_a_cuda), compute_dist(a, q_a_cpu))

if __name__ == "__main__":
    unittest.main()
