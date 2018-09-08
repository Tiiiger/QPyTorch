#include <cuda.h>

/**
 * quantize real number in [a] into fixed point number in [b] with minimal unit
 * [sigma], and minimum number [t_min], and maximum number [t_max],
 * using stochastic rounding with random number in [r]
 **/
__global__ void fixed_point_quantize_copy_kernel(float* __restrict__ a,
                                                 float* __restrict__ r,
                                                 float* o, int size, float sigma,
                                                 float t_min, float t_max);

__global__ void extract_max_exponent_kernel(float *a, short *o, int size);

__global__ void reduce_max_exponent_kernel(short *a, short *o, int size);
