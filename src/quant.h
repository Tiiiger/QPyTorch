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
Tensor block_quantize_cuda(Tensor a, Tensor r, int wl);
