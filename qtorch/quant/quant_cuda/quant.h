#include <ATen/ATen.h>

using namespace at;

/**
 * quantize a FloatTensor into fixed point number with word length [wl]
 * and fractional bits [fl].
 * Stochastic Rounding.
 **/
Tensor fixed_point_quantize_stochastic_cuda(Tensor a, Tensor r, int wl, int fl);

/**
 * quantize a FloatTensor into fixed point number with word length [wl]
 * and fractional bits [fl]
 * Nearest Rounding.
 **/
Tensor fixed_point_quantize_nearest_cuda(Tensor a, int wl, int fl);

/**
 * quantize a FloatTensor into fixed point number with word length [wl]
 * and fractional bits [fl]
 * Stochastic Rounding.
 **/
Tensor block_quantize_stochastic_aten_cuda(Tensor a, Tensor r, int wl);

/**
 * quantize a FloatTensor into fixed point number with word length [wl]
 * and fractional bits [fl]
 * Nearest Rounding.
 **/
Tensor block_quantize_nearest_aten_cuda(Tensor a, int wl);

/**
 * quantize a FloatTensor into a low bit-width floating point Tensor
 * with [man_bits] mantissa bits and [exp_bits] exponent bits.
 * Does not handle NaN, Inf, and denormal.
 * Stochastic Rounding.
 **/
Tensor float_quantize_stochastic_cuda(Tensor a, int man_bits, int exp_bits);

/**
 * quantize a FloatTensor into a low bit-width floating point Tensor
 * with [man_bits] mantissa bits and [exp_bits] exponent bits.
 * Does not handle NaN, Inf, and denormal.
 * Nearest Rounding.
 **/
Tensor float_quantize_nearest_cuda(Tensor a, int man_bits, int exp_bits);
