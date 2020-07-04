import torch
import qtorch
from qtorch.quant import posit_quantize
#full_precision_tensor = torch.rand(5).cuda()
import numpy as np
a = np.array([-15.0])
full_precision_tensor = torch.tensor(a,dtype=torch.float)
print("Full Precision: {}".format(full_precision_tensor))
low_precision_tensor = posit_quantize(full_precision_tensor, nsize=4, es=1, rounding="nearest")
print("Low Precision: {}".format(low_precision_tensor))
