#include "quant_kernel.h"
#include <cmath>

__device__ __forceinline__ float round_helper(float a, float r) {
  // return floor(a+r);
  return nearbyint(a+r-0.5);
}

__device__ __forceinline__ float round(float a, float r, int sigma) {
  a = ldexp(a, -sigma); 
  a = round_helper(a, r);
  a = ldexp(a, sigma);
  return a;
}

__device__ __forceinline__ float nearest_round(float a, int sigma) {
  a = ldexp(a, -sigma); 
  // a = nearbyint(a);
  a = round(a);
  // a = floor(a+0.5);
  //a = ceil(a-0.5);
  a = ldexp(a, sigma);
  return a;
}
