#include <ATen/ATen.h>

using namespace at;

/**
 * quantize a FloatTensor into fixed point number with word length [wl]
 * and fractional bits [fl]
 **/
Tensor fixed_point_quantize_cuda(Tensor a, Tensor r, int wl, int fl);

/**
 * quantize a FloatTensor into fixed point number with word length [wl]
 * and fractional bits [fl]
 **/
Tensor block_quantize_aten_cuda(Tensor a, Tensor r, int wl);

/**
 * quantize a FloatTensor into a low bit-width floating point Tensor
 * with [man_bits] mantissa bits and [exp_bits] exponent bits.
 * Does not handle NaN, Inf, and denormal.
 **/
Tensor float_quantize_cuda(Tensor a, Tensor r, int man_bits, int exp_bits);
