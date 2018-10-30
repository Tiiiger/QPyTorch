#include <torch/torch.h>
#include <assert.h>
// #include <cstdlib>
// #include <time.h>
#include <random>

using namespace at;

#define CHECK_CONTIGUOUS(x) assert(x.is_contiguous())
#define CHECK_DEVICE(x) assert(x.device() == kCPU)
#define CHECK_INPUT(x) CHECK_DEVICE(x)
#define RFLOAT_TO_BITS(x) (*reinterpret_cast<unsigned int*>(x))
#define RBITS_TO_FLOAT(x) (*reinterpret_cast<float*>(x))
#define FLOAT_TO_BITS(f, i) assert(sizeof f == sizeof i); std::memcpy(&i, &f, sizeof i)
#define BITS_TO_FLOAT(i, f) assert(sizeof f == sizeof i); std::memcpy(&f, &i, sizeof f)

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<> dis(0);

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

Tensor fixed_point_quantize_stochastic(Tensor a, int wl, int fl) {
  CHECK_INPUT(a);
  auto r = rand_like(a);
  auto a_array = a.data<float>();
  auto r_array = r.data<float>();
  Tensor o = zeros_like(a);
  auto o_array = o.data<float>();
  int64_t size = a.numel();
  int sigma = -fl;
  float t_min = -ldexp(1.0, wl-fl-1);
  float t_max = -t_min-ldexp(1.0, sigma);
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

enum Mode {rNearest, rStochastic};

unsigned int round_bitwise(unsigned int target, int man_bits, Mode rounding){
  unsigned int mask = (1 << (23-man_bits)) - 1;
  unsigned int rand_prob;
  if (rounding == rStochastic) {
    rand_prob = (dis(gen)) & mask;
  } else {
    rand_prob = 1 << (23-man_bits-1);
  }
  unsigned int add_r = target+rand_prob;
  unsigned int quantized = add_r & ~mask;
  return quantized;
}

void block_quantize_helper(float* input, float* output, float max_elem,
                           int wl, int size, Mode rounding) {
  unsigned int max_num;
  FLOAT_TO_BITS(max_elem, max_num);
  unsigned int max_exp = max_num << 1 >> 24 << 23;
  float base_float;
  BITS_TO_FLOAT(max_exp, base_float);
  base_float *= 6;

  for (int64_t i=0; i < size; i++) {
    float target_rebase = input[i]+base_float;
    unsigned int target_bits;
    FLOAT_TO_BITS(target_rebase, target_bits);
    unsigned int quantized_bits = round_bitwise(target_bits, wl, rounding); // -1 sign, -1 virtual, +2 base
    float quantized_rebase;
    BITS_TO_FLOAT(quantized_bits, quantized_rebase);
    float quantized = quantized_rebase-base_float;
    output[i] = quantized;
  }
}

Tensor block_quantize_nearest(Tensor a, int wl) {
  CHECK_INPUT(a);
  auto a_array = a.data<float>();
  Tensor o = zeros_like(a);
  auto o_array = o.data<float>();
  int64_t size = a.numel();

  // get maximum number and base
  Tensor max_entry = at::max(at::abs(a));
  auto max_elem = max_entry.data<float>();
  block_quantize_helper(a_array, o_array, *max_elem, wl, size, rNearest);
  return o;
}

Tensor block_quantize_stochastic(Tensor a, int wl) {
  CHECK_INPUT(a);
  auto a_array = a.data<float>();
  Tensor o = zeros_like(a);
  auto o_array = o.data<float>();
  int64_t size = a.numel();

  // get maximum number and base
  Tensor max_entry = at::max(at::abs(a));
  auto max_elem = max_entry.data<float>();
  // std::srand(time(0));
  block_quantize_helper(a_array, o_array, *max_elem, wl, size, rStochastic);
  return o;
}

unsigned int clip_exponent(unsigned int target, int exp_bits) {
  return target;
}

//TODO: DRAFT, NEED TO SLEEP
Tensor float_quantize_stochastic(Tensor a, int man_bits, int exp_bits) {
  // use external random number right now
  auto a_array = a.data<float>();
  auto o = zeros_like(a);
  auto o_array = o.data<float>();
  int size = a.numel();

  for (int64_t i=0; i < size; i++) {
    man_bits = man_bits-1;
    unsigned int target;
    FLOAT_TO_BITS(a_array[i], target);
    round_bitwise(target, man_bits, rStochastic);
    target = clip_exponent(target, exp_bits);
    float quantized;
    BITS_TO_FLOAT(target, quantized);
    o_array[i] = quantized;
  }
  return o;
}

Tensor float_quantize_nearest(Tensor a, int man_bits, int exp_bits) {
  auto a_array = a.data<float>();
  auto o = zeros_like(a);
  auto o_array = o.data<float>();
  int size = a.numel();

  for (int64_t i=0; i < size; i++) {
    man_bits = man_bits-1;
    unsigned int target;
    FLOAT_TO_BITS(a_array[i], target);
    round_bitwise(target, man_bits, rNearest);
    target = clip_exponent(target, exp_bits);
    float quantized;
    BITS_TO_FLOAT(target, quantized);
    o_array[i] = quantized;
  }
  return o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fixed_point_quantize_stochastic", &fixed_point_quantize_stochastic, "Fixed Point Number Stochastic Quantization (CPU)");
  m.def("block_quantize_stochastic", &block_quantize_stochastic, "Block Floating Point Number Stochastic Quantization (CPU)");
  m.def("float_quantize_stochastic", &float_quantize_stochastic, "Low-Bitwidth Floating Point Number Stochastic Quantization (CUDA)");
  m.def("fixed_point_quantize_nearest", &fixed_point_quantize_nearest, "Fixed Point Number Nearest Neighbor Quantization (CPU)");
  m.def("block_quantize_nearest", &block_quantize_nearest, "Block Floating Point Number Nearest Neighbor Quantization (CPU)");
  m.def("float_quantize_nearest", &float_quantize_nearest, "Low-Bitwidth Floating Point Number Nearest Neighbor Quantization (CPU)");
}


