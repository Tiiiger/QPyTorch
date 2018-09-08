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

__device__ __forceinline__ float clamp_helper(float a, float min, float max) {
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



__global__ void fixed_point_quantize_inplace_kernel(float *a,  float* __restrict__ r, int size,
                                     float sigma, float t_min, float t_max) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    a[index] = stochastic_round(a[index], r[index], sigma);
    a[index] = clamp_helper(a[index], t_min, t_max);
  }
}

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

__global__ void block_quantize_copy_kernel(float* __restrict__ a,
                                           float* __restrict__ r,
                                           float* o, int size, int wl,
                                           short *exponent) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    int sigma = (int) exponent[0]-(wl-1);
    o[index] = stochastic_round(a[index], r[index], sigma);
  }
}

Tensor fixed_point_quantize_cuda(Tensor a, Tensor r, int wl, int fl) {
  // use external random number right now
  auto o = at::zeros_like(a);
  auto dim = a.dim();
  int64_t size = 1;
  for (int i=0; i<dim; i++) size *=a.size(i);
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