#include <torch/torch.h>
#include <assert.h>
#include <random>
#include <tuple>
#include "quant_cpu.h"


using namespace at;

enum Mode
{
  rNearest,
  rStochastic
};

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_CPU(x) TORCH_CHECK(!x.is_cuda(), #x " must be a CPU tensor")
#define CHECK_INPUT(x) \
  CHECK_CPU(x);        \
  CHECK_CONTIGUOUS(x);

#define RFLOAT_TO_BITS(x) (*reinterpret_cast<uint64_t *>(x))
#define RBITS_TO_FLOAT(x) (*reinterpret_cast<float *>(x))
#define FLOAT_TO_BITS(f, i)     \
  assert(sizeof f == sizeof i); \
  std::memcpy(&i, &f, sizeof i)
#define BITS_TO_FLOAT(i, f)     \
  assert(sizeof f == sizeof i); \
  std::memcpy(&f, &i, sizeof f)

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<> dis(0);

template <typename T>
T clamp_helper(T a, T min, T max)
{
  if (a > max)
    return max;
  else if (a < min)
    return min;
  else
    return a;
}

void printBits(size_t const size, void const * const ptr)
{
    unsigned char *b = (unsigned char*) ptr;
    unsigned char byte;
    int64_t i, j;
    
    for (i = size-1; i >= 0; i--) {
        for (j = 7; j >= 0; j--) {

            byte = (b[i] >> j) & 1;
            printf("%u", byte);
            if ((i==size-1 && j==7) || (i==size-2 && j==7))
              printf(" ");
        }
    }
}

template <typename T>
T clamp_mask_helper(T a, T min, T max, uint8_t *mask)
{
  if (a > max)
  {
    *mask = 1;
    return max;
  }
  else if (a < min)
  {
    *mask = 1;
    return min;
  }
  else
    return a;
}

std::tuple<Tensor, Tensor> fixed_point_quantize_stochastic_mask(Tensor a, int64_t wl, int64_t fl, bool symmetric)
{
  CHECK_INPUT(a);
  auto r = rand_like(a);
  auto a_array = a.data_ptr<float>();
  auto r_array = r.data_ptr<float>();
  auto o = zeros_like(a);
  auto o_array = o.data_ptr<float>();
  auto m = zeros_like(a, torch::CPU(kByte));
  auto m_array = m.data_ptr<uint8_t>();
  int64_t size = a.numel();
  int sigma = -fl;
  float t_min, t_max;
  fixed_min_max(wl, fl, symmetric, &t_min, &t_max);
  for (int64_t i = 0; i < size; i++)
  {
    o_array[i] = round(a_array[i], r_array[i], sigma);
    o_array[i] = clamp_mask_helper<float>(o_array[i], t_min, t_max, m_array + i);
  }
  return std::make_tuple(o, m);
}

std::tuple<Tensor, Tensor> fixed_point_quantize_nearest_mask(Tensor a, int64_t wl, int64_t fl, bool symmetric)
{
  CHECK_INPUT(a);
  auto a_array = a.data_ptr<float>();
  auto o = zeros_like(a);
  auto o_array = o.data_ptr<float>();
  auto m = zeros_like(a, torch::CPU(kByte));
  auto m_array = m.data_ptr<uint8_t>();
  int64_t size = a.numel();
  int sigma = -fl;
  float t_min, t_max;
  fixed_min_max(wl, fl, symmetric, &t_min, &t_max);
  for (int64_t i = 0; i < size; i++)
  {
    o_array[i] = round(a_array[i], 0.5, sigma);
    o_array[i] = clamp_mask_helper<float>(o_array[i], t_min, t_max, m_array + i);
  }
  return std::make_tuple(o, m);
}

Tensor fixed_point_quantize_stochastic(Tensor a, int64_t wl, int64_t fl, bool clamp, bool symmetric)
{
  CHECK_INPUT(a);
  auto r = rand_like(a);
  auto a_array = a.data_ptr<float>();
  auto r_array = r.data_ptr<float>();
  Tensor o = zeros_like(a);
  auto o_array = o.data_ptr<float>();
  int64_t size = a.numel();
  int sigma = -fl;
  float t_min, t_max;
  fixed_min_max(wl, fl, symmetric, &t_min, &t_max);
  for (int64_t i = 0; i < size; i++)
  {
    o_array[i] = round(a_array[i], r_array[i], sigma);
    if (clamp)
    {
      o_array[i] = clamp_helper(o_array[i], t_min, t_max);
    }
  }
  return o;
}

Tensor fixed_point_quantize_nearest(Tensor a, int64_t wl, int64_t fl, bool clamp, bool symmetric)
{
  CHECK_INPUT(a);
  auto a_array = a.data_ptr<float>();
  Tensor o = zeros_like(a);
  auto o_array = o.data_ptr<float>();
  int64_t size = a.numel();
  int sigma = -fl;
  float t_min, t_max;
  fixed_min_max(wl, fl, symmetric, &t_min, &t_max);
  for (int64_t i = 0; i < size; i++)
  {
    o_array[i] = round(a_array[i], 0.5, sigma);
    if (clamp)
    {
      o_array[i] = clamp_helper(o_array[i], t_min, t_max);
    }
  }
  return o;
}

uint64_t round_bitwise(uint64_t target, int64_t man_bits, Mode rounding)
{
  
  uint64_t mask = (1 << (23 - man_bits)) - 1;
  uint64_t rand_prob;
  if (rounding == rStochastic)
  {
    rand_prob = (dis(gen)) & mask;
  }
  else
  {
    rand_prob = 1 << (23 - man_bits - 1);
  }
  uint64_t add_r = target + rand_prob;
  uint64_t quantized = add_r & ~mask;
  return quantized;
}

void block_quantize_helper(float *input, float *output, float *max_elem,
                           int64_t wl, int64_t size, Mode rounding)
{
  for (int64_t i = 0; i < size; i++)
  {

    unsigned int max_num;
    FLOAT_TO_BITS(max_elem[i], max_num);
    unsigned int max_exp = max_num << 1 >> 24 << 23;
    float base_float;
    BITS_TO_FLOAT(max_exp, base_float);
    base_float *= 6;

    float target_rebase = input[i] + base_float;
    unsigned int target_bits;
    FLOAT_TO_BITS(target_rebase, target_bits);
    unsigned int quantized_bits = round_bitwise(target_bits, wl, rounding); // -1 sign, -1 virtual, +2 base
    float quantized_rebase;
    BITS_TO_FLOAT(quantized_bits, quantized_rebase);
    float quantized = quantized_rebase - base_float;

    unsigned int quantize_bits;
    FLOAT_TO_BITS(quantized, quantize_bits);
    unsigned int clip_quantize = clip_max_exponent(wl - 2, max_exp, quantize_bits);
    BITS_TO_FLOAT(clip_quantize, quantized);

    output[i] = quantized;
  }
}

Tensor get_max_entry(Tensor a, int64_t dim)
{
  Tensor max_entry;
  if (dim == -1)
  {
    max_entry = at::max(at::abs(a)).expand_as(a).contiguous();
  }
  else if (dim == 0)
  {
    Tensor input_view = a.view({a.size(0), -1});
    max_entry = std::get<0>(input_view.abs().max(1, true)).expand_as(input_view).view_as(a).contiguous();
  }
  else
  {
    Tensor input_transpose = a.transpose(0, dim);
    Tensor input_view = input_transpose.contiguous().view({input_transpose.size(0), -1});
    Tensor max_transpose = std::get<0>(input_view.abs().max(1, true)).expand_as(input_view).view_as(input_transpose);
    max_entry = max_transpose.transpose(dim, 0).contiguous();
  }
  return max_entry;
}

Tensor block_quantize_nearest(Tensor a, int64_t wl, int64_t dim)
{
  CHECK_INPUT(a);
  auto a_array = a.data_ptr<float>();
  Tensor o = zeros_like(a);
  auto o_array = o.data_ptr<float>();
  int64_t size = a.numel();

  // get maximum number and base
  Tensor max_entry = get_max_entry(a, dim);
  auto max_elem = max_entry.data_ptr<float>();
  block_quantize_helper(a_array, o_array, max_elem, wl, size, rNearest);
  return o;
}

Tensor block_quantize_stochastic(Tensor a, int64_t wl, int64_t dim)
{
  CHECK_INPUT(a);
  auto a_array = a.data_ptr<float>();
  Tensor o = zeros_like(a);
  auto o_array = o.data_ptr<float>();
  int64_t size = a.numel();

  // get maximum number and base
  Tensor max_entry = get_max_entry(a, dim);
  auto max_elem = max_entry.data_ptr<float>();
  // std::srand(time(0));
  block_quantize_helper(a_array, o_array, max_elem, wl, size, rStochastic);
  return o;
}

Tensor float_quantize(Tensor a, int64_t man_bits, int64_t exp_bits, Mode rounding){
  auto a_array = a.data_ptr<float>();
  auto o = zeros_like(a);
  auto o_array = o.data_ptr<float>();
  int64_t size = a.numel();

  for (int64_t i = 0; i < size; i++)
  {
    uint64_t target,quantize_bits;
    FLOAT_TO_BITS(a_array[i], target);
    float quantized;

    int64_t target_exp = (target << 1 >> 1 >> 23) -127; 
    int64_t min_exp = -((1 << (exp_bits - 1)) - 2);
    bool subnormal = (target_exp < min_exp);
    if (subnormal){
      float shift_float,val;
      int64_t shift_bits = ((127+min_exp)<<23) | (target >> 31 <<31);
      BITS_TO_FLOAT(shift_bits, shift_float);
      val=a_array[i]+shift_float;
      FLOAT_TO_BITS(val, target);
      quantize_bits = round_bitwise(target, man_bits, rounding);
      BITS_TO_FLOAT(quantize_bits, quantized);
      quantized=quantized-shift_float;
    }
    else{
      quantize_bits = round_bitwise(target, man_bits, rounding);
      quantize_bits = clip_exponent(exp_bits, man_bits, target, quantize_bits);
      BITS_TO_FLOAT(quantize_bits, quantized);
    }
    o_array[i] = quantized;
  }
  return o;
}

Tensor float_quantize_stochastic(Tensor a, int64_t man_bits, int64_t exp_bits)
{
  return float_quantize(a,man_bits, exp_bits, rStochastic);
}

Tensor float_quantize_nearest(Tensor a, int64_t man_bits, int64_t exp_bits)
{
  return float_quantize(a,man_bits, exp_bits, rNearest);
}

TORCH_LIBRARY(qtorch_ops, m) {
  m.def("fixed_point_quantize_stochastic_mask", fixed_point_quantize_stochastic_mask);
  m.def("fixed_point_quantize_stochastic", fixed_point_quantize_stochastic);
  m.def("block_quantize_stochastic", block_quantize_stochastic);
  m.def("float_quantize_stochastic", float_quantize_stochastic);
  m.def("fixed_point_quantize_nearest_mask", fixed_point_quantize_nearest_mask);
  m.def("fixed_point_quantize_nearest", fixed_point_quantize_nearest);
  m.def("block_quantize_nearest", block_quantize_nearest);
  m.def("float_quantize_nearest", float_quantize_nearest);
}
