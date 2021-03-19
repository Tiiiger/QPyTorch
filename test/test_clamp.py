import torch
import unittest
from qtorch.quant import *
from qtorch import FixedPoint, BlockFloatingPoint, FloatingPoint

DEBUG = False

log = lambda m: print(m) if DEBUG else False

class TestStochastic(unittest.TestCase):
    """
    invariant: quantized numbers cannot be greater than the maximum representable number
    or lower than the maximum representable number
    """

    def test_fixed(self):
        """test fixed point clamping"""
        for d in ["cpu", "cuda"]:
            for r in ["stochastic", "nearest"]:
                wl = 5
                fl = 4
                t_min = -(2 ** (wl - fl - 1))
                t_max = 2 ** (wl - fl - 1) - 2 ** (-fl)
                a = torch.linspace(-2, 2, steps=100, device=d)
                clamp_a = fixed_point_quantize(a, wl=wl, fl=fl, clamp=True, rounding=r)
                self.assertEqual(t_max, clamp_a.max().item())
                self.assertEqual(t_min, clamp_a.min().item())

                a = torch.linspace(-2, 2, steps=100, device=d)
                no_clamp_a = fixed_point_quantize(a, wl=wl, fl=fl, clamp=False, rounding=r)
                self.assertLess(t_max, no_clamp_a.max().item())
                self.assertGreater(t_min, no_clamp_a.min().item())

    def test_float(self):
        """test floating point quantization"""
        formats = [(2,2),(2,3),(3,2)]

        for exp, man in formats:
            for d in ["cpu", "cuda"]:
                for r in ["stochastic", "nearest"]:
                    a_max = 2 ** (2 ** (exp - 1)) * (1 - 2 ** (-man - 1))
                    a_min = 2 ** (-(2 ** (exp - 1)) + 1)
                    max_exp=int((2**exp)/2)
                    min_exp=-(max_exp-2)
                    mantissa_step=2**(-man)
                    min_mantissa=mantissa_step   # When denormalized
                    max_mantissa=2-mantissa_step # When normalized, mantissa goes from 1 to 2-mantissa_step
                    a_min = 2**min_exp*min_mantissa
                    a_max = 2**max_exp*max_mantissa
                    expected_vals=[]
                    log(f"With {exp} exponent bits, our exponent goes from {min_exp} to {max_exp}")
                    log(f"With {man} mantissa bits, our mantissa goes from {min_mantissa} (denormalized) to {max_mantissa}")
                    log(f"With {man} mantissa bits and {exp} exponent bits, we can go from {a_min} to {a_max}")
                    representable_normalized =[]
                    for sign in [1,-1]:
                        for e in range(0,2**exp):
                            for m in range(0,2**man):
                                if e==0:
                                    val = sign*(2**(e+min_exp)*m*2**(-man))
                                    log(f"{0 if sign==1 else 1} {e:0{exp}b} {m:0{man}b} = {sign} * 2^{e+min_exp} * {m*2**(-man)} \t= {val} (denormalized)")
                                else:
                                    val = sign*(2**(e+min_exp-1)*(1+(m*2**(-man))))
                                    log(f"{0 if sign==1 else 1} {e:0{exp}b} {m:0{man}b} = {sign} * 2^{e+min_exp-1} * {1+(m*2**(-man))} \t= {val}")
                                if val not in expected_vals:
                                    expected_vals.append(val)
                    expected_vals.sort()
                    
                    # Block box test to get representable numbers
                    import numpy as np
                    quant_vals=[]
                    for i in np.arange(-30,30,.01):
                        a = torch.Tensor([i]).to(device=d)
                        quant_a = float_quantize(a, exp=exp, man=man, rounding=r)
                        if quant_a[0] not in quant_vals:
                            quant_vals.append(quant_a[0].item())
                    log("Values representable in QPytorch")
                    log(quant_vals)

                    self.assertEqual(quant_vals, expected_vals)

if __name__ == "__main__":
    unittest.main()
