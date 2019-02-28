import torch
import unittest
import qtorch
from qtorch.quant import block_quantize, fixed_point_quantize, float_quantize
from qtorch import FixedPoint, BlockFloatingPoint, FloatingPoint

class TestDevice(unittest.TestCase):
    """
        invariant: cuda and cpp implementation should behave the same
    """
    def error(self, cuda_t, cpu_t):
        return ((cuda_t.cpu()-cpu_t)**2).sum().item()

    def test_fixed_point(self):
        for wl, fl in [(5,4), (3,2)]:
            for rounding in ["nearest"]:
                t_max = 1-(2**(-fl))
                to_quantize_cuda = torch.linspace(-t_max, t_max, steps=20, device='cuda')
                to_quantize_cpu = to_quantize_cuda.clone().to("cpu")
                fixed_quantized_cuda = fixed_point_quantize(to_quantize_cuda, wl=wl, fl=fl, rounding=rounding)
                fixed_quantized_cpu = fixed_point_quantize(to_quantize_cpu, wl=wl, fl=fl, rounding=rounding)
                mse = self.error(fixed_quantized_cuda, fixed_quantized_cpu)
                self.assertTrue(mse<1e-15)
                # self.assertTrue(torch.eq(fixed_quantized_cuda.cpu(), fixed_quantized_cpu).all().item())

    def test_block_floating_point(self):
        for wl in [5, 3]:
            for rounding in ["nearest"]:
                for dim in [-1, 0, 1]:
                    t_max = 1-(2**(-4))
                    to_quantize_cuda = torch.linspace(-t_max, t_max, steps=20, device='cuda')
                    to_quantize_cpu = to_quantize_cuda.clone().to("cpu")
                    block_quantized_cuda = block_quantize(to_quantize_cuda, wl=wl, rounding=rounding)
                    block_quantized_cpu =  block_quantize(to_quantize_cpu,  wl=wl, rounding=rounding)
                    mse = self.error(block_quantized_cuda, block_quantized_cpu)
                    self.assertTrue(mse<1e-15)
                    # self.assertTrue(torch.eq(block_quantized_cuda.cpu(), block_quantized_cpu).all().item())

    def test_floating_point(self):
        for man, exp in [(2, 5), (6, 9)]:
            for rounding in ["nearest"]:
                to_quantize_cuda = torch.rand(20).cuda()
                to_quantize_cpu = to_quantize_cuda.clone().to("cpu")
                float_quantized_cuda = float_quantize(to_quantize_cuda, man=man, exp=exp, rounding=rounding)
                float_quantized_cpu  = float_quantize(to_quantize_cpu,  man=man, exp=exp, rounding=rounding)
                mse = self.error(float_quantized_cuda, float_quantized_cpu)
                self.assertTrue(mse<1e-15)

if __name__ == "__main__":
    unittest.main()
