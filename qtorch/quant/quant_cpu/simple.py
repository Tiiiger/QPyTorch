import torch
import quant_cpu

a = torch.randn(3, 3)*2
wl = 2
fl = 1
b, m = quant_cpu.fixed_point_quantize_nearest_mask(a, wl, fl, True)
print(a, b, m)
