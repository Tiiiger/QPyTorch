#include "quant_kernel.h"

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