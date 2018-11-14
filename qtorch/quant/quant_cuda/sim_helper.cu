#include "quant_kernel.h"
#include <cmath>

__device__ __forceinline__ float round_helper(float a, float r) {
  return floor(a+r);
}

__device__ __forceinline__ float round(float a, float r, int sigma) {
  a = ldexp(a, -sigma); 
  a = round_helper(a, r);
  a = ldexp(a, sigma);
  return a;
}

__device__ __forceinline__ float nearest_round(float a, int sigma) {
  a = ldexp(a, -sigma); 
  a = round(a);
  // a = round_helper(a, 0.5);
  a = ldexp(a, sigma);
  return a;
}