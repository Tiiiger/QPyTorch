import torch
import qtorch
from qtorch.quant import posit_quantize, configurable_table_quantize
#full_precision_tensor = torch.rand(5).cuda()
import numpy as np
a = np.random.rand(10) * 20-10
table = np.array(range(-10,10))
table_tensor = torch.tensor(table,dtype=torch.float)
full_precision_tensor = torch.tensor(a,dtype=torch.float)
print("Full Precision: {}".format(full_precision_tensor))
low_precision_tensor = posit_quantize(full_precision_tensor, nsize=4, es=1, rounding="nearest")
print("Low Precision P(4,1): {}".format(low_precision_tensor))

low_precision_tensor = posit_quantize(full_precision_tensor.cuda(), nsize=4, es=1, rounding="nearest")
print("Low Precision P(4,1) CUDA: {}".format(low_precision_tensor))

low_precision_tensor = posit_quantize(full_precision_tensor, nsize=5, es=1, rounding="nearest")
print("Low Precision P(5,1): {}".format(low_precision_tensor))

low_precision_tensor = posit_quantize(full_precision_tensor.cuda(), nsize=5, es=1, rounding="nearest")
print("Low Precision P(5,1) CUDA: {}".format(low_precision_tensor))


low_precision_tensor = posit_quantize(full_precision_tensor, nsize=15, es=2, rounding="nearest")
print("Low Precision P(15,2): {}".format(low_precision_tensor))

low_precision_tensor = posit_quantize(full_precision_tensor.cuda(), nsize=15, es=2, rounding="nearest")
print("Low Precision P(15,2) CUDA: {}".format(low_precision_tensor))

low_precision_tensor = posit_quantize(full_precision_tensor, nsize=7, es=2, rounding="nearest")
print("Low Precision P(7,2): {}".format(low_precision_tensor))

low_precision_tensor = posit_quantize(full_precision_tensor.cuda(), nsize=7, es=2, rounding="nearest")
print("Low Precision P(7,2) CUDA: {}".format(low_precision_tensor))
'''
low_precision_tensor = configurable_table_quantize(full_precision_tensor, table_tensor)
print(": TableLookup sample {}".format(low_precision_tensor))
low_precision_tensor = configurable_table_quantize(full_precision_tensor.cuda(), table_tensor)
print(": TableLookup sample CUDA {}".format(low_precision_tensor))
'''
