import torch
import qtorch
from qtorch.quant import posit_quantize
full_precision_tensor = torch.rand(5).cuda()
print("Full Precision: {}".format(full_precision_tensor))
low_precision_tensor = posit_quantize(full_precision_tensor, nsize=8, es=2, rounding="nearest")
print("Low Precision: {}".format(low_precision_tensor))
