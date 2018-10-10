#include <torch/torch.h>
#include <assert.h>

using namespace at;

#define CHECK_CONTIGUOUS(x) assert(x.is_contiguous())
#define CHECK_INPUT(x) CHECK_CONTIGUOUS(x)

float stochastic_round_helper(float a, float r) {
  return floor(a+r);
}

template <typename T>
T clamp_helper(T a, T min, T max) {
  if (a > max) return max;
  else if (a < min) return min;
  else return a;
}

float stochastic_round(float a, float r, int sigma) {
  a = ldexp(a, -sigma); 
  a = stochastic_round_helper(a, r);
  a = ldexp(a, sigma);
  return a;
}

unsigned int extract_exponent(float *a) {
  unsigned int temp = *(reinterpret_cast<unsigned int*>(a));
  temp = (temp << 1 >> 24); // single precision, 1 sign bit, 23 mantissa bits
  return temp-127+1; // exponent offset and virtual bit
}

Tensor fixed_point_quantize(Tensor a, Tensor r, int wl, int fl) {
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
    o_array[i] = stochastic_round(a_array[i], r_array[i], sigma);
    o_array[i] = clamp_helper(o_array[i], t_min, t_max);
  }
  return o;
}

Tensor block_quantize(Tensor a, Tensor r, int wl) {
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
    o_array[i] = stochastic_round(a_array[i], r_array[i], sigma);
  }
  return o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fixed_point_quantize", &fixed_point_quantize, "Fixed Point Number Quantization (CPU)");
  m.def("block_quantize", &block_quantize, "Block Floating Point Number Quantization (CPU)");
  //m.def("float_quantize", &float_quantize, "Low-Bitwidth Floating Point Number Quantization (CUDA)");
}


