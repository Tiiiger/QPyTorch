/**
 * kernel that quantize real numbers into fixed point numbers.
 **/
__global__ void fixed_point_quantize_copy_cuda(float* __restrict__ a,
                                               float* __restrict__ r,
                                               float* o, int size, float sigma,
                                               float t_min, float t_max);


