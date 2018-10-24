#include <torch/torch.h>
#include "quant.h"

using namespace at;

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

Tensor fixed_point_quantize_nearest(Tensor a, int wl, int fl) {
  CHECK_INPUT(a);
  return fixed_point_quantize_nearest_cuda(a, wl, fl);
}

Tensor block_quantize_nearest(Tensor a, int wl) {
  CHECK_INPUT(a);
  return block_quantize_nearest_cuda(a, wl);
}

Tensor float_quantize_nearest(Tensor a, int man_bits, int exp_bits) {
  CHECK_INPUT(a);
  return float_quantize_nearest_cuda(a, man_bits, exp_bits);
}

Tensor fixed_point_quantize_stochastic(Tensor a, Tensor r, int wl, int fl) {
  CHECK_INPUT(a);
  CHECK_INPUT(r);
  return fixed_point_quantize_stochastic_cuda(a, r, wl, fl);
}

Tensor block_quantize_stochastic(Tensor a, int wl) {
  CHECK_INPUT(a);
  return block_quantize_stochastic_cuda(a, wl);
}

Tensor float_quantize_stochastic(Tensor a, int man_bits, int exp_bits) {
  CHECK_INPUT(a);
  return float_quantize_stochastic_cuda(a, man_bits, exp_bits);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fixed_point_quantize_stochastic", &fixed_point_quantize_stochastic, "Fixed Point Number Stochastic Quantization (CUDA)");
  m.def("block_quantize_stochastic", &block_quantize_stochastic, "Block Floating Point Number Stochastic Quantization (CUDA)");
  m.def("float_quantize_stochastic", &float_quantize_stochastic, "Low-Bitwidth Floating Point Number Stochastic Quantization (CUDA)");
  m.def("fixed_point_quantize_nearest", &fixed_point_quantize_nearest, "Fixed Point Number Nearest Neighbor Quantization (CUDA)");
  m.def("block_quantize_nearest", &block_quantize_nearest, "Block Floating Point Number Nearest Neighbor Quantization (CUDA)");
  m.def("float_quantize_nearest", &float_quantize_nearest, "Low-Bitwidth Floating Point Number Nearest Neighbor Quantization (CUDA)");
}
