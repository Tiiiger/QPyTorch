#include <cstdlib>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <climits>
#include <ATen/ATen.h>

#define FLOAT_TO_BITS(x) (*reinterpret_cast<unsigned int*>(x))
#define BITS_TO_FLOAT(x) (*reinterpret_cast<float*>(x))

using namespace at;

__device__ __forceinline__ float round_helper(float a, float r) {
  return floor(a+r);
}

template <typename T>
__device__ __forceinline__ T clamp_helper(T a, T min, T max) {
  if (a > max) return max;
  else if (a < min) return min;
  else return a;
}

__device__ __forceinline__ float round(float a, float r, int sigma) {
  a = ldexp(a, -sigma); 
  a = round_helper(a, r);
  a = ldexp(a, sigma);
  return a;
}

// quantize an array of real numbers into fixed point with word length [wl] and [fl] fractional bits
// 2**-[sigma] is the smallest unit of the fixed point representation. Stochastic Rounding with r.
__global__ void fixed_point_quantize_copy_kernel_stochastic(float* __restrict__ a,
                                                            float* __restrict__ r,
                                                            float* o, int size, int sigma,
                                                            float t_min, float t_max) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    o[index] = round(a[index], r[index], sigma);
    o[index] = clamp_helper(o[index], t_min, t_max);
  }
}

// quantize an array of real numbers into fixed point with word length [wl] and [fl] fractional bits
// 2**-[sigma] is the smallest unit of the fixed point representation. Nearest Neighbor Rounding.
__global__ void fixed_point_quantize_copy_kernel_nearest(float* __restrict__ a,
                                                         float* o, int size, int sigma,
                                                         float t_min, float t_max) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    o[index] = round(a[index], 0.5, sigma);
    o[index] = clamp_helper(o[index], t_min, t_max);
  }
}

__device__ __forceinline__ unsigned int extract_exponent(float *a) {
  unsigned int temp = *(reinterpret_cast<unsigned int*>(a));
  temp = (temp << 1 >> 24); // single preciision, 1 sign bit, 23 mantissa bits
  return temp-127+1; // exponent offset and virtual bit
}

__device__ __forceinline__ unsigned int round_bitwise_stochastic(unsigned int target,
                                                                 unsigned int rand_prob,
                                                                 int man_bits) {
    unsigned int mask = (1 << (23-man_bits)) - 1;
    unsigned int add_r = target+(rand_prob & mask);
    unsigned int quantized = add_r & ~mask;
    return quantized;
}

__device__ __forceinline__ unsigned int round_bitwise_nearest(unsigned int target,
                                                              int man_bits) {
    unsigned int mask = (1 << (23-man_bits)) - 1;
    unsigned int rand_prob = 1 << (23-man_bits-1);
    unsigned int add_r = target+rand_prob;
    unsigned int quantized = add_r & ~mask;
    return quantized;
}

__device__ __forceinline__ unsigned int clip_exponent(int exp_bits, int man_bits,
                                                      unsigned int old_num,
                                                      unsigned int quantized_num) {
  // int offset = 32-9-man_bits; // float length minus sign bit and exponent bit add 1 virtual bit
  // unsigned int quantized_exponent_store = quantized_num << 1 >> 1 >> 23; // 1 sign bit, 23 mantissa bits
  // int quantized_exponent_real = (int) quantized_exponent_store - 126;
  // unsigned int max_exponent = (unsigned int) 1 << exp_bits;
  // if (quantized_exponent > max_exponent) {
  //   unsigned int max_man = (unsigned int ) -1 << (32-wl) >> 9; // 1 sign bit, 8 exponent bits
  //   unsigned int max_num = (max_exponent << 23) | max_man;
  //   unsigned int old_sign = old_num >> 31 << 31;
  //   quantized_num = old_sign | max_num;
  // }
  // return quantized_num;
}

// quantize a float into a floating point with [exp_bits] exponent and
// [man_bits] mantissa
__global__ void float_kernel_stochastic(float* __restrict__ a,
                                        int* __restrict__ r,
                                        float* o, int size,
                                        int man_bits,
                                        int exp_bits) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    man_bits = man_bits-1; // 1 virtual bit
    unsigned int old_num = FLOAT_TO_BITS(&a[index]);
    unsigned int rand_prob = (unsigned int) r[index];
    unsigned int quantize = round_bitwise_stochastic(old_num, rand_prob, man_bits);
    // quantize = clip_exponent(exp_bits, man_bits, old_num, quantize);
    float quantize_float = BITS_TO_FLOAT(&quantize);
    o[index] = quantize_float;
  }
}

// quantize a float into a floating point with [exp_bits] exponent and
// [man_bits] mantissa
__global__ void float_kernel_nearest(float* __restrict__ a,
                                     float* o, int size,
                                     int man_bits,
                                     int exp_bits) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    man_bits = man_bits-1; // 1 virtual bit
    unsigned int old_num = FLOAT_TO_BITS(&a[index]);
    unsigned int quantize = round_bitwise_nearest(old_num, man_bits);
    // quantize = clip_exponent(exp_bits, man_bits, old_num, quantize);
    float quantize_float = BITS_TO_FLOAT(&quantize);
    o[index] = quantize_float;
  }
}

// quantize a float into a floating point with [exp_bits] exponent and
// [man_bits] mantissa
__global__ void block_kernel_stochastic(float* __restrict__ a,
                                        int* __restrict__ r,
                                        float* o, int size,
                                        float* max_entry,
                                        int man_bits) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    unsigned int max_entry_bits = FLOAT_TO_BITS(max_entry);
    unsigned int max_exp = max_entry_bits << 1 >> 24 << 23;
    float base_float = 6*BITS_TO_FLOAT(&max_exp);

    float target_rebase = a[index]+base_float;
    unsigned int target_bits = FLOAT_TO_BITS(&target_rebase);
    unsigned int rand_prob = (unsigned int) r[index];
    unsigned int quantized = round_bitwise_stochastic(target_bits, rand_prob, man_bits);
    float quantize_float = BITS_TO_FLOAT(&quantized)-base_float;
    o[index] = quantize_float;
  }
}

Tensor block_quantize_stochastic_cuda(Tensor a, int wl) {
  auto o = at::zeros_like(a);
  auto rand_ints = randint_like(a, INT_MAX, device(kCUDA).dtype(kInt));
  int64_t size = a.numel();

  Tensor max_entry = at::max(at::abs(a));
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  block_kernel_stochastic<<<blockNums, blockSize>>>(a.data<float>(),
                                                    rand_ints.data<int>(),
                                                    o.data<float>(),
                                                    size,
                                                    max_entry.data<float>(),
                                                    wl);
  return o;
}

// quantize a float into a floating point with [exp_bits] exponent and
// [man_bits] mantissa
__global__ void block_kernel_nearest(float* __restrict__ a,
                                        float* o, int size,
                                        float* max_entry,
                                        int man_bits) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    unsigned int max_entry_bits = FLOAT_TO_BITS(max_entry);
    unsigned int max_exp = max_entry_bits << 1 >> 24 << 23;
    float base_float = 6*BITS_TO_FLOAT(&max_exp);

    float target_rebase = a[index]+base_float;
    unsigned int target_bits = FLOAT_TO_BITS(&target_rebase);
    unsigned int quantized = round_bitwise_nearest(target_bits, man_bits);
    float quantize_float = BITS_TO_FLOAT(&quantized)-base_float;
    o[index] = quantize_float;
  }
}

Tensor block_quantize_nearest_cuda(Tensor a, int wl) {
  auto o = at::zeros_like(a);
  auto rand_ints = randint_like(a, INT_MAX, device(kCUDA).dtype(kInt));
  int64_t size = a.numel();

  Tensor max_entry = at::max(at::abs(a));
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  block_kernel_nearest<<<blockNums, blockSize>>>(a.data<float>(),
                                                 o.data<float>(),
                                                 size,
                                                 max_entry.data<float>(),
                                                 wl);
  return o;
}

Tensor float_quantize_stochastic_cuda(Tensor a, int man_bits, int exp_bits) {
  // use external random number right now
  auto o = zeros_like(a);
  auto rand_ints = randint_like(a, INT_MAX, device(kCUDA).dtype(kInt));
  int size = a.numel();
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  float_kernel_stochastic<<<blockNums, blockSize>>>(a.data<float>(),
                                                    rand_ints.data<int>(),
                                                    o.data<float>(),
                                                    size,
                                                    man_bits,
                                                    exp_bits);
  return o;
}

Tensor float_quantize_nearest_cuda(Tensor a, int man_bits, int exp_bits) {
  // use external random number right now
  auto o = zeros_like(a);
  int size = a.numel();
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  float_kernel_nearest<<<blockNums, blockSize>>>(a.data<float>(),
                                                 o.data<float>(),
                                                 size,
                                                 man_bits,
                                                 exp_bits);
  return o;
}

Tensor fixed_point_quantize_stochastic_cuda(Tensor a, Tensor r, int wl, int fl) {
  // use external random number right now
  auto o = at::zeros_like(a);
  int64_t size = a.numel();
  int sigma = -fl;
  float t_min = -ldexp(1.0, wl-fl-1);
  float t_max = -t_min-ldexp(1.0, sigma);
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  fixed_point_quantize_copy_kernel_stochastic<<<blockNums, blockSize>>>(a.data<float>(),
                                                                        r.data<float>(),
                                                                        o.data<float>(),
                                                                        size,
                                                                        sigma,
                                                                        t_min,
                                                                        t_max);
  return o;
}

Tensor fixed_point_quantize_nearest_cuda(Tensor a, int wl, int fl) {
  // use external random number right now
  auto o = at::zeros_like(a);
  int64_t size = a.numel();
  int sigma = -fl;
  float t_min = -ldexp(1.0, wl-fl-1);
  float t_max = -t_min-ldexp(1.0, sigma);
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  fixed_point_quantize_copy_kernel_nearest<<<blockNums, blockSize>>>(a.data<float>(),
                                                                     o.data<float>(),
                                                                     size,
                                                                     sigma,
                                                                     t_min,
                                                                     t_max);
  return o;
}
