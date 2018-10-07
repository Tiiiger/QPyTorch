#include <iostream>
#include <cstdlib>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include "quant_kernel.h"

using namespace at;

__device__ __forceinline__ float stochastic_round_helper(float a, float r) {
  return floor(a+r);
}

template <typename T>
__device__ __forceinline__ T clamp_helper(T a, T min, T max) {
  if (a > max) return max;
  else if (a < min) return min;
  else return a;
}

__device__ __forceinline__ float stochastic_round(float a, float r, int sigma) {
  a = ldexp(a, -sigma); 
  a = stochastic_round_helper(a, r);
  a = ldexp(a, sigma);
  return a;
}

// quantize an array of real numbers into fixed point with word length [wl] and [fl] fractional bits
// 2**-[sigma] is the smallest unit of the fixed point representation
__global__ void fixed_point_quantize_copy_kernel(float* __restrict__ a,
                                                 float* __restrict__ r,
                                                 float* o, int size, int sigma,
                                                 float t_min, float t_max) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    o[index] = stochastic_round(a[index], r[index], sigma);
    o[index] = clamp_helper(o[index], t_min, t_max);
  }
}

__device__ __forceinline__ unsigned int extract_exponent(float *a) {
  unsigned int temp = *(reinterpret_cast<unsigned int*>(a));
  temp = (temp << 1 >> 24); // single preciision, 1 sign bit, 23 mantissa bits
  return temp-127+1; // exponent offset and virtual bit
}

// quantize an array of real number into block floating point
// each number has word length [wl] and [max_entry] is the maximum number
// in array
__global__ void block_quantize_copy_aten_kernel(float* __restrict__ a,
                                                float* __restrict__ r,
                                                float* o, int size, int wl,
                                                float *max_entry) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    int exponent = ((int) extract_exponent(max_entry);
    int sigma = exponent-(wl-1);
    o[index] = stochastic_round(a[index], r[index], sigma);
  }
}

// quantize a float into a floating point with [exp_bits] exponent and
// [man_bits] mantissa
// TODO: Need Testing
__global__ void float_kernel(float* __restrict__ a,
                             int* __restrict__ r,
                             float* o, int size,
                             int exp_bits,
                             int man_bits) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    man_bits = man_bits-1; // 1 virtual bit
    unsigned int max_exponent = (unsigned int) -1 << (32-exp_bits) >> (32-exp_bits);

    unsigned int old_number = *reinterpret_cast<unsigned int*>(a[index]);
    unsigned int old_exponent = old_number << 1 >> 1 >> 23;
    unsigned int rand_prob = (unsigned int) r[index];
    unsigned int add_r = old_number+rand_prob;
    int offset = 32-9-man_bits; // float length minus sign bit and exponent bit add 1 virtual bit
    unsigned int mask = (unsigned int) -1 << offset;
    unsigned int quantize = add_r & mask;
    unsigned int quantized_exponent = quantize << 1 >> 1 >> 23; // 1 sign bit, 23 mantissa bits
    if (quantized_exponent > max_exponent) {
      unsigned int max_man = (unsigned int ) -1 << 9 >> 9 >> offset << offset; // 23 mantissa bits, 1 virtual bit
      unsigned int max_num = (max_exponent << 23) | max_man;
      unsigned int old_sign = old_number >> 31 << 31;
      quantize = old_sign | max_num;
    }
    o[index] = quantize;
  }
}

int64_t calc_size(Tensor a) {
  auto dim = a.dim();
  int64_t size = 1;
  for (int i=0; i<dim; i++) size *=a.size(i);
  return size;
}

Tensor float_quantize_cuda(Tensor a, Tensor r, int man_bits, int exp_bits) {
  // use external random number right now
  auto o = at::zeros_like(a);
  int64_t size = calc_size(a);
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  float_kernel<<<blockNums, blockSize>>>(a.data<float>(),
                                         r.data<float>(),
                                         o.data<float>(),
                                         size,
                                         man_bits,
                                         exp_bits);
  return o;
}

Tensor fixed_point_quantize_cuda(Tensor a, Tensor r, int wl, int fl) {
  // use external random number right now
  auto o = at::zeros_like(a);
  int64_t size = calc_size(a);
  int sigma = -fl;
  float t_min = -ldexp(1.0, wl-fl-1);
  float t_max = -t_min-sigma;
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  fixed_point_quantize_copy_kernel<<<blockNums, blockSize>>>(a.data<float>(),
                                                             r.data<float>(),
                                                             o.data<float>(),
                                                             size,
                                                             sigma,
                                                             t_min,
                                                             t_max);
  return o;
}

Tensor block_quantize_aten_cuda(Tensor a, Tensor r, int wl) {
  auto o = at::zeros_like(a);
  int64_t size = calc_size(a);

  Tensor max_entry = at::max(at::abs(a));
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  block_quantize_copy_aten_kernel<<<blockNums, blockSize>>>(a.data<float>(),
                                                            r.data<float>(),
                                                            o.data<float>(),
                                                            size,
                                                            wl,
                                                            max_entry.data<float>());
  return o;

}

Tensor block_quantize_cuda(Tensor a, Tensor r, int wl) {
  auto o = at::zeros_like(a);
  auto dim = a.dim();
  int64_t size = 1;
  for (int i=0; i<dim; i++) size *=a.size(i);

  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  short *temp; 
  cudaMalloc(&temp, blockNums*sizeof(short));

  extract_max_exponent_kernel<<<blockNums, blockSize>>>(a.data<float>(),
                                                        temp,
                                                        size);
  reduce_max_exponent_kernel<<<1, 1024>>>(temp,
                                          temp,
                                          blockNums);

  block_quantize_copy_kernel<<<blockNums, blockSize>>>(a.data<float>(),
                                                       r.data<float>(),
                                                       o.data<float>(),
                                                       size,
                                                       wl,
                                                       temp);
  cudaFree(temp);
  return o;
}