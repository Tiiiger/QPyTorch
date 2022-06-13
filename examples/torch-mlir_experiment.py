import torch
import torch.nn as nn

import qtorch
# from qtorch.quant import block_quantize
torch.ops.load_library("qtorch/quant/quant_cpu/build/libqtorch_ops.so")
print(torch.ops.qtorch_ops.block_quantize_nearest)

import torch_mlir
from torch_mlir_e2e_test.tosa_backends import linalg_on_tensors

class SimpleModel(nn.Module):
    def __init__(self, input_dim, output_size):
        super(SimpleModel, self).__init__()
        self.matmul = nn.Linear(input_dim, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        matmul_out = self.matmul(x.flatten(1))
        quantized_matmul_out = torch.ops.qtorch_ops.block_quantize_nearest(matmul_out, 8, 0)
        relu_out = self.relu(quantized_matmul_out)
        return relu_out

batches = 5
input_dim = 64
output_size = 4
inputs = torch.randn(batches, input_dim)
model = SimpleModel(input_dim, output_size)
print("forward propagate results on inputs is:\n", model.forward(inputs))

# quantized_inputs = block_quantize(inputs, wl=8, dim=0, rounding="nearest")
# print("forward propagate of quantized inputs result is ", model.forward(quantized_inputs))

module = torch_mlir.compile(model, inputs, output_type=torch_mlir.OutputType.TOSA, use_tracing=False)
print("Module compiled to TOSA is:\n", module)
