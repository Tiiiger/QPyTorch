#include "quant_cpu.h"
#include <math.h>
#include <stdint.h>

void fixed_min_max(int wl, int fl, bool symmetric, float *t_min, float *t_max)
{
  int sigma = -fl;
  *t_min = -ldexp(1.0, wl - fl - 1);
  *t_max = -*t_min - ldexp(1.0, sigma);
  if (symmetric)
    *t_min = *t_min + ldexp(1.0, sigma);
}

float round(float a, float r, int sigma)
{
  a = ldexp(a, -sigma);
  a = nearbyint(a + r - 0.5);
  // a = floor(a + r);
  a = ldexp(a, sigma);
  return a;
}
