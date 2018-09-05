#include <torch/torch.h>
#include "quant.h"

using namespace at;

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

Tensor fixed_point_quantize(Tensor a, Tensor r, int wl, int fl) {
  CHECK_INPUT(a);
  CHECK_INPUT(r);
  return fixed_point_quantize_cuda(a, r, wl, fl);
}


Tensor block_quantize(Tensor a, Tensor r, Tensor temp, int wl) {
  CHECK_INPUT(a);
  CHECK_INPUT(r);
  CHECK_INPUT(temp);
  return block_quantize_cuda(a, r, temp, wl);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fixed_point_quantize", &fixed_point_quantize, "Fixed Point Number Quantization (CUDA)");
  m.def("block_quantize", &block_quantize, "Block Floating Point Number Quantization (CUDA)");
}
