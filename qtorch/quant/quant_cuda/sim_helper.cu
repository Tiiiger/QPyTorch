#include "quant_kernel.h"

__device__ __forceinline__ float round_helper(float a, float r) {
  return floor(a+r);
}

__device__ __forceinline__ float round(float a, float r, int sigma) {
  a = ldexp(a, -sigma); 
  a = round_helper(a, r);
  a = ldexp(a, sigma);
  return a;
}