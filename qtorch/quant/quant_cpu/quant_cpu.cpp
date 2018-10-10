#include <torch/torch.h>
#include <assert.h>

using namespace at;

#define CHECK_CONTIGUOUS(x) assert(x.is_contiguous())
#define CHECK_INPUT(x) CHECK_CONTIGUOUS(x)

template <typename T>
T clamp_helper(T a, T min, T max) {
  if (a > max) return max;
  else if (a < min) return min;
  else return a;
}

float round(float a, float r, int sigma) {
  a = ldexp(a, -sigma); 
  a = floor(a+r);
  a = ldexp(a, sigma);
  return a;
}

unsigned int extract_exponent(float *a) {
  unsigned int temp = *(reinterpret_cast<unsigned int*>(a));
  temp = (temp << 1 >> 24); // single precision, 1 sign bit, 23 mantissa bits
  return temp-127+1; // exponent offset and virtual bit
}

Tensor fixed_point_quantize_stochastic(Tensor a, Tensor r, int wl, int fl) {
  CHECK_INPUT(a);
  CHECK_INPUT(r);
  auto a_array = a.data<float>();
  auto r_array = r.data<float>();
  Tensor o = zeros_like(a);
  auto o_array = o.data<float>();
  int64_t size = a.numel();
  int sigma = -fl;
  float t_min = -ldexp(1.0, wl-fl-1);
  float t_max = -t_min-sigma;
  for (int64_t i=0; i < size; i++) {
    o_array[i] = round(a_array[i], r_array[i], sigma);
    o_array[i] = clamp_helper(o_array[i], t_min, t_max);
  }
  return o;
}

Tensor fixed_point_quantize_nearest(Tensor a, int wl, int fl) {
  CHECK_INPUT(a);
  auto a_array = a.data<float>();
  Tensor o = zeros_like(a);
  auto o_array = o.data<float>();
  int64_t size = a.numel();
  int sigma = -fl;
  float t_min = -ldexp(1.0, wl-fl-1);
  float t_max = -t_min-sigma;
  for (int64_t i=0; i < size; i++) {
    o_array[i] = round(a_array[i], 0.5, sigma);
    o_array[i] = clamp_helper(o_array[i], t_min, t_max);
  }
  return o;
}

Tensor block_quantize_stochastic(Tensor a, Tensor r, int wl) {
  CHECK_INPUT(a);
  CHECK_INPUT(r);
  auto a_array = a.data<float>();
  auto r_array = r.data<float>();
  Tensor o = zeros_like(a);
  auto o_array = o.data<float>();
  int64_t size = a.numel();

  Tensor max_entry = at::max(at::abs(a));
  // auto max_entry = max_tensor.data<float>();
  auto max_elem = max_entry.data<float>();
  int exponent = ((int) extract_exponent(max_elem));
  int sigma = exponent-(wl-1);
  
  for (int64_t i=0; i < size; i++) {
    o_array[i] = round(a_array[i], r_array[i], sigma);
  }
  return o;
}

Tensor block_quantize_nearest(Tensor a, int wl) {
  CHECK_INPUT(a);
  auto a_array = a.data<float>();
  Tensor o = zeros_like(a);
  auto o_array = o.data<float>();
  int64_t size = a.numel();

  Tensor max_entry = at::max(at::abs(a));
  // auto max_entry = max_tensor.data<float>();
  auto max_elem = max_entry.data<float>();
  int exponent = ((int) extract_exponent(max_elem));
  int sigma = exponent-(wl-1);
  
  for (int64_t i=0; i < size; i++) {
    o_array[i] = round(a_array[i], 0.5, sigma);
  }
  return o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fixed_point_quantize_stochastic", &fixed_point_quantize_stochastic, "Fixed Point Number Stochastic Quantization (CPU)");
  m.def("block_quantize_stochastic", &block_quantize_stochastic, "Block Floating Point Number Stochastic Quantization (CPU)");
  //m.def("float_quantize_stochastic", &float_quantize_stochastic, "Low-Bitwidth Floating Point Number Stochastic Quantization (CUDA)");
  m.def("fixed_point_quantize_nearest", &fixed_point_quantize_nearest, "Fixed Point Number Nearest Neighbor Quantization (CPU)");
  m.def("block_quantize_nearest", &block_quantize_nearest, "Block Floating Point Number Nearest Neighbor Quantization (CPU)");
  // m.def("float_quantize_nearest", &float_quantize_nearest, "Low-Bitwidth Floating Point Number Nearest Neighbor Quantization (CPU)");
}


