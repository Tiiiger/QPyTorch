#include "quant_cpu.h"
#include <math.h>
#include <stdint.h>



template <typename T>
T clamp_helper(T a, T min, T max) {
  if (a > max) return max;
  else if (a < min) return min;
  else return a;
}

template <typename T>
T clamp_mask_helper(T a, T min, T max, uint8_t* mask) {
  if (a > max) {
    *mask = 1;
    return max;
  }
  else if (a < min) {
    *mask = 1;
    return min;
  }
  else return a;
}

void fixed_min_max(int wl, int fl, bool symmetric, float* t_min, float* t_max) {
  int sigma = -fl;
  *t_min = -ldexp(1.0, wl-fl-1);
  *t_max = -*t_min-ldexp(1.0, sigma);
  if (symmetric) *t_min = *t_min+ldexp(1.0, sigma);
}

float round(float a, float r, int sigma) {
  a = ldexp(a, -sigma);
  a = floor(a+r);
  a = ldexp(a, sigma);
  return a;
}
