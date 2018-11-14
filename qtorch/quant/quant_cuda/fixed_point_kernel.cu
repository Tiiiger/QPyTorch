#include "quant_kernel.h"
#include "sim_helper.cu"


template <typename T>
__device__ __forceinline__ T clamp_helper(T a, T min, T max) {
  if (a > max) return max;
  else if (a < min) return min;
  else return a;
}

template <typename T>
__device__ __forceinline__ T clamp_mask_helper(T a, T min, T max, uint8_t* mask) {
  if (a > max) {
    *mask = 1;
    return max;
  } else if (a < min) {
    *mask = 1;
    return min;
  }
  *mask = 0;
  return a;
}

// quantize an array of real numbers into fixed point with word length [wl] and [fl] fractional bits
// 2**-[sigma] is the smallest unit of the fixed point representation. Stochastic Rounding with r.
__global__ void fixed_point_quantize_kernel_stochastic(float* __restrict__ a,
                                                       float* __restrict__ r,
                                                       float* o, int size,
                                                       int sigma, bool use_clamp,
                                                       float t_min, float t_max) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    o[index] = round(a[index], r[index], sigma);
    if (use_clamp) {
      o[index] = clamp_helper(o[index], t_min, t_max);
    }
  }
}

// quantize an array of real numbers into fixed point with word length [wl] and [fl] fractional bits
// 2**-[sigma] is the smallest unit of the fixed point representation. Nearest Neighbor Rounding.
__global__ void fixed_point_quantize_kernel_nearest(float* __restrict__ a,
                                                    float* o, int size,
                                                    int sigma, bool use_clamp,
                                                    float t_min, float t_max) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    o[index] = nearest_round(a[index], sigma);
    if (use_clamp) {
      o[index] = clamp_helper(o[index], t_min, t_max);
    }
  }
}

__global__ void fixed_point_quantize_kernel_mask_stochastic(float* __restrict__ a,
                                                            float* __restrict__ r,
                                                            float* o, uint8_t* m,
                                                            int size, int sigma,
                                                            float t_min, float t_max) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    o[index] = round(a[index], r[index], sigma);
    o[index] = clamp_mask_helper(o[index], t_min, t_max, m+index);
  }
}

__global__ void fixed_point_quantize_kernel_mask_nearest(float* __restrict__ a,
                                                         float* o, uint8_t* m,
                                                         int size, int sigma,
                                                         float t_min, float t_max) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    o[index] = nearest_round(a[index], sigma);
    o[index] = clamp_mask_helper(o[index], t_min, t_max, m+index);
  }
}