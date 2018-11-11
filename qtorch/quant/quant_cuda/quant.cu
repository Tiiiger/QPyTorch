#include <cstdlib>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <climits>
#include <ATen/ATen.h>
#include "quant_cuda.h"
#include "quant_kernel.h"

using namespace at;

Tensor block_quantize_stochastic_cuda(Tensor a, int wl) {
  auto o = at::zeros_like(a);
  auto rand_ints = randint_like(a, INT_MAX, device(kCUDA).dtype(kInt));
  int64_t size = a.numel();

  Tensor max_entry = at::max(at::abs(a));
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  block_kernel_stochastic<<<blockNums, blockSize>>>(a.data<float>(),
                                                    rand_ints.data<int>(),
                                                    o.data<float>(),
                                                    size,
                                                    max_entry.data<float>(),
                                                    wl);
  return o;
}

Tensor block_quantize_nearest_cuda(Tensor a, int wl) {
  auto o = at::zeros_like(a);
  int64_t size = a.numel();

  Tensor max_entry = at::max(at::abs(a));
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  block_kernel_nearest<<<blockNums, blockSize>>>(a.data<float>(),
                                                 o.data<float>(),
                                                 size,
                                                 max_entry.data<float>(),
                                                 wl);
  return o;
}

Tensor block_quantize_sim_stochastic_cuda(Tensor a, int wl) {
  auto o = at::zeros_like(a);
  auto rand_probs = rand_like(a);
  int64_t size = a.numel();

  Tensor max_entry = at::max(at::abs(a));
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  block_kernel_sim_stochastic<<<blockNums, blockSize>>>(a.data<float>(),
                                                        rand_probs.data<float>(),
                                                        o.data<float>(),
                                                        size,
                                                        max_entry.data<float>(),
                                                        wl);
  return o;
}

Tensor block_quantize_sim_nearest_cuda(Tensor a, int wl) {
  auto o = at::zeros_like(a);
  auto rand_ints = randint_like(a, INT_MAX, device(kCUDA).dtype(kInt));
  int64_t size = a.numel();

  Tensor max_entry = at::max(at::abs(a));
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  block_kernel_sim_nearest<<<blockNums, blockSize>>>(a.data<float>(),
                                                 o.data<float>(),
                                                 size,
                                                 max_entry.data<float>(),
                                                 wl);
  return o;
}

Tensor float_quantize_stochastic_cuda(Tensor a, int man_bits, int exp_bits) {
  // use external random number right now
  auto o = zeros_like(a);
  auto rand_ints = randint_like(a, INT_MAX, device(kCUDA).dtype(kInt));
  int size = a.numel();
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  float_kernel_stochastic<<<blockNums, blockSize>>>(a.data<float>(),
                                                    rand_ints.data<int>(),
                                                    o.data<float>(),
                                                    size,
                                                    man_bits,
                                                    exp_bits);
  return o;
}

Tensor float_quantize_nearest_cuda(Tensor a, int man_bits, int exp_bits) {
  // use external random number right now
  auto o = zeros_like(a);
  int size = a.numel();
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  float_kernel_nearest<<<blockNums, blockSize>>>(a.data<float>(),
                                                 o.data<float>(),
                                                 size,
                                                 man_bits,
                                                 exp_bits);
  return o;
}

Tensor fixed_point_quantize_stochastic_cuda(Tensor a, int wl, int fl) {
  // use external random number right now
  auto o = at::zeros_like(a);
  auto rand_probs = rand_like(a);
  int64_t size = a.numel();
  int sigma = -fl;
  float t_min = -ldexp(1.0, wl-fl-1);
  float t_max = -t_min-ldexp(1.0, sigma);
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  fixed_point_quantize_copy_kernel_stochastic<<<blockNums, blockSize>>>(a.data<float>(),
                                                                        rand_probs.data<float>(),
                                                                        o.data<float>(),
                                                                        size,
                                                                        sigma,
                                                                        t_min,
                                                                        t_max);
  return o;
}

Tensor fixed_point_quantize_nearest_cuda(Tensor a, int wl, int fl) {
  // use external random number right now
  auto o = at::zeros_like(a);
  int64_t size = a.numel();
  int sigma = -fl;
  float t_min = -ldexp(1.0, wl-fl-1);
  float t_max = -t_min-ldexp(1.0, sigma);
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  fixed_point_quantize_copy_kernel_nearest<<<blockNums, blockSize>>>(a.data<float>(),
                                                                     o.data<float>(),
                                                                     size,
                                                                     sigma,
                                                                     t_min,
                                                                     t_max);
  return o;
}