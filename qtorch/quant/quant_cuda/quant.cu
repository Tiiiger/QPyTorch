#include "quant_cuda.h"
#include "quant_kernel.h"
#include <ATen/ATen.h>
#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>
#include <stdint.h>
#include <tuple>

using namespace at;

__device__ int subOptIwl(float *__restrict__ a, int len) {
  int max = 0, iwl = 0;
  for (int i = 0; i < len; i++) {
    int n = a[i];
    if (n < 0)
      n *= (-1);
    if (max < n)
      max = n;
  }
  if (max)
    iwl = log2(max) + 1;
  return iwl;
}
__global__ void optIwl(float *__restrict__ a, int len, int *res) {
  *res = subOptIwl(a, len);
}

Tensor get_max_entry(Tensor a, int dim) {
  Tensor max_entry;
  if (dim == -1) {
    max_entry = at::max(at::abs(a)).expand_as(a).contiguous();
  } else if (dim == 0) {
    Tensor input_view = a.view({a.size(0), -1});
    max_entry = std::get<0>(input_view.abs().max(1, true))
                    .expand_as(input_view)
                    .view_as(a)
                    .contiguous();
  } else {
    Tensor input_transpose = a.transpose(0, dim);
    Tensor input_view =
        input_transpose.contiguous().view({input_transpose.size(0), -1});
    Tensor max_transpose = std::get<0>(input_view.abs().max(1, true))
                               .expand_as(input_view)
                               .view_as(input_transpose);
    max_entry = max_transpose.transpose(dim, 0).contiguous();
  }
  return max_entry;
}

Tensor block_quantize_stochastic_cuda(Tensor a, int wl, int dim) {
  auto o = at::zeros_like(a);
  auto rand_ints = randint_like(a, INT_MAX, device(kCUDA).dtype(kInt));
  int64_t size = a.numel();

  Tensor max_entry = get_max_entry(a, dim);
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  block_kernel_stochastic<<<blockNums, blockSize>>>(
      a.data_ptr<float>(), rand_ints.data_ptr<int>(), o.data_ptr<float>(), size,
      max_entry.data_ptr<float>(), wl);
  return o;
}

Tensor block_quantize_nearest_cuda(Tensor a, int wl, int dim) {
  auto o = at::zeros_like(a);
  int64_t size = a.numel();

  Tensor max_entry = get_max_entry(a, dim);
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  block_kernel_nearest<<<blockNums, blockSize>>>(
      a.data_ptr<float>(), o.data_ptr<float>(), size,
      max_entry.data_ptr<float>(), wl);
  return o;
}

Tensor block_quantize_sim_stochastic_cuda(Tensor a, int wl) {
  auto o = at::zeros_like(a);
  auto rand_probs = rand_like(a);
  int64_t size = a.numel();

  Tensor max_entry = at::max(at::abs(a));
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  block_kernel_sim_stochastic<<<blockNums, blockSize>>>(
      a.data_ptr<float>(), rand_probs.data_ptr<float>(), o.data_ptr<float>(),
      size, max_entry.data_ptr<float>(), wl);
  return o;
}

Tensor block_quantize_sim_nearest_cuda(Tensor a, int wl) {
  auto o = at::zeros_like(a);
  auto rand_ints = randint_like(a, INT_MAX, device(kCUDA).dtype(kInt));
  int64_t size = a.numel();

  Tensor max_entry = at::max(at::abs(a));
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  block_kernel_sim_nearest<<<blockNums, blockSize>>>(
      a.data_ptr<float>(), o.data_ptr<float>(), size,
      max_entry.data_ptr<float>(), wl);
  return o;
}

Tensor float_quantize_stochastic_cuda(Tensor a, int man_bits, int exp_bits) {
  // use external random number right now
  auto o = zeros_like(a);
  auto rand_ints = randint_like(a, INT_MAX, device(kCUDA).dtype(kInt));
  int size = a.numel();
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  float_kernel_stochastic<<<blockNums, blockSize>>>(
      a.data_ptr<float>(), rand_ints.data_ptr<int>(), o.data_ptr<float>(), size,
      man_bits, exp_bits);
  return o;
}

Tensor float_quantize_nearest_cuda(Tensor a, int man_bits, int exp_bits) {
  // use external random number right now
  auto o = zeros_like(a);
  int size = a.numel();
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  float_kernel_nearest<<<blockNums, blockSize>>>(
      a.data_ptr<float>(), o.data_ptr<float>(), size, man_bits, exp_bits);
  return o;
}

void fixed_min_max(int wl, int fl, bool symmetric, float *t_min, float *t_max) {
  int sigma = -fl;
  *t_min = -ldexp(1.0, wl - fl - 1);
  *t_max = -*t_min - ldexp(1.0, sigma);
  if (symmetric)
    *t_min = *t_min + ldexp(1.0, sigma);
}

Tensor fixed_point_quantize_stochastic_cuda(Tensor a, int wl, int fl,
                                            bool use_clamp, bool symmetric,
                                            bool dynamic_precision) {
  // use external random number right now
  auto o = at::zeros_like(a);
  auto rand_probs = rand_like(a);
  int64_t size = a.numel();
  int sigma;
  float t_min, t_max;
  if (dynamic_precision) {
    int res;
    int *dev_c;
    cudaMalloc((void **)&dev_c, sizeof(int));
    optIwl<<<1, 1>>>(a.data_ptr<float>(), size, dev_c);
    cudaMemcpy(&res, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
    fl = wl - res - 1;
    if (fl < 0)
      fl = 0;
  }
  sigma = -fl;
  fixed_min_max(wl, fl, symmetric, &t_min, &t_max);
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;
  fixed_point_quantize_kernel_stochastic<<<blockNums, blockSize>>>(
      a.data_ptr<float>(), rand_probs.data_ptr<float>(), o.data_ptr<float>(),
      size, sigma, use_clamp, t_min, t_max);
  return o;
}

Tensor fixed_point_quantize_nearest_cuda(Tensor a, int wl, int fl,
                                         bool use_clamp, bool symmetric,
                                         bool dynamic_precision) {
  // use external random number right now
  auto o = at::zeros_like(a);
  int64_t size = a.numel();
  int sigma;
  float t_min, t_max;
  if (dynamic_precision) {
    int res;
    int *dev_c;
    cudaMalloc((void **)&dev_c, sizeof(int));
    optIwl<<<1, 1>>>(a.data_ptr<float>(), size, dev_c);
    cudaMemcpy(&res, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
    fl = wl - res - 1;
    if (fl < 0)
      fl = 0;
  }
  sigma = -fl;
  fixed_min_max(wl, fl, symmetric, &t_min, &t_max);
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  fixed_point_quantize_kernel_nearest<<<blockNums, blockSize>>>(
      a.data_ptr<float>(), o.data_ptr<float>(), size, sigma, use_clamp, t_min,
      t_max);
  return o;
}

std::tuple<Tensor, Tensor> fixed_point_quantize_stochastic_mask_cuda(
    Tensor a, int wl, int fl, bool symmetric, bool dynamic_precision) {
  // use external random number right now
  auto o = zeros_like(a);
  auto rand_probs = rand_like(a);
  auto m = zeros_like(a, a.options().dtype(kByte));
  int64_t size = a.numel();
  int sigma;
  float t_min, t_max;
  if (dynamic_precision) {
    int res;
    int *dev_c;
    cudaMalloc((void **)&dev_c, sizeof(int));
    optIwl<<<1, 1>>>(a.data_ptr<float>(), size, dev_c);
    cudaMemcpy(&res, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
    fl = wl - res - 1;
    if (fl < 0)
      fl = 0;
  }
  sigma = -fl;
  fixed_min_max(wl, fl, symmetric, &t_min, &t_max);
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  fixed_point_quantize_kernel_mask_stochastic<<<blockNums, blockSize>>>(
      a.data_ptr<float>(), rand_probs.data_ptr<float>(), o.data_ptr<float>(),
      m.data_ptr<uint8_t>(), size, sigma, t_min, t_max);
  return std::make_tuple(o, m);
}

std::tuple<Tensor, Tensor>
fixed_point_quantize_nearest_mask_cuda(Tensor a, int wl, int fl, bool symmetric,
                                       bool dynamic_precision) {
  // use external random number right now
  auto o = at::zeros_like(a);
  auto m = zeros_like(a, a.options().dtype(kByte));
  int64_t size = a.numel();
  int sigma;
  float t_min, t_max;
  if (dynamic_precision) {
    int res;
    int *dev_c;
    cudaMalloc((void **)&dev_c, sizeof(int));
    optIwl<<<1, 1>>>(a.data_ptr<float>(), size, dev_c);
    cudaMemcpy(&res, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
    fl = wl - res - 1;
    if (fl < 0)
      fl = 0;
  }
  sigma = -fl;
  fixed_min_max(wl, fl, symmetric, &t_min, &t_max);
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  fixed_point_quantize_kernel_mask_nearest<<<blockNums, blockSize>>>(
      a.data_ptr<float>(), o.data_ptr<float>(), m.data_ptr<uint8_t>(), size,
      sigma, t_min, t_max);
  return std::make_tuple(o, m);
}