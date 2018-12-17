#include "quant_kernel.h"
#include "bit_helper.cu"

// quantize a float into a floating point with [exp_bits] exponent and
// [man_bits] mantissa
__global__ void float_kernel_stochastic(float* __restrict__ a,
                                        int* __restrict__ r,
                                        float* o, int size,
                                        int man_bits,
                                        int exp_bits) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    unsigned int old_num = FLOAT_TO_BITS(&a[index]);
    unsigned int rand_prob = (unsigned int) r[index];
    unsigned int quantize = round_bitwise_stochastic(old_num, rand_prob, man_bits);
    quantize = clip_exponent(exp_bits, man_bits, old_num, quantize);
    float quantize_float = BITS_TO_FLOAT(&quantize);
    o[index] = quantize_float;
  }
}

// quantize a float into a floating point with [exp_bits] exponent and
// [man_bits] mantissa
__global__ void float_kernel_nearest(float* __restrict__ a,
                                     float* o, int size,
                                     int man_bits,
                                     int exp_bits) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    unsigned int old_num = FLOAT_TO_BITS(&a[index]);
    unsigned int quantize = round_bitwise_nearest(old_num, man_bits);
    quantize = clip_exponent(exp_bits, man_bits, old_num, quantize);
    float quantize_float = BITS_TO_FLOAT(&quantize);
    o[index] = quantize_float;
  }
}