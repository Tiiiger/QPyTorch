#include <iostream>
#include <cstdlib>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>

using namespace at;

__device__ __forceinline__ float stochastic_round_helper(float a, float r) {
  return floor(a+r);
}

__device__ __forceinline__ float clamp_helper(float a, float min, float max) {
  if (a > max) return max;
  else if (a < min) return min;
  else return a;
}

__device__ __forceinline__ float stochastic_round(float a, float r, float sigma) {
  a /= sigma; 
  a = stochastic_round_helper(a, r);
  a *= sigma;
  return a;
}



__global__ void fixed_point_quantize_inplace_kernel(float *a,  float* __restrict__ r, int size,
                                     float sigma, float t_min, float t_max) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    a[index] = stochastic_round(a[index], r[index], sigma);
    a[index] = clamp_helper(a[index], t_min, t_max);
  }
}

__global__ void fixed_point_quantize_copy_kernel(float* __restrict__ a,
                                                  float* __restrict__ r,
                                                  float* o, int size, float sigma,
                                                  float t_min, float t_max) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    o[index] = stochastic_round(a[index], r[index], sigma);
    o[index] = clamp_helper(o[index], t_min, t_max);
  }
}

Tensor fixed_point_quantize_cuda(Tensor a, Tensor r, int wl, int fl) {
  // use external random number right now
  auto o = at::zeros_like(a);
  auto dim = a.dim();
  int64_t size = 1;
  for (int i=0; i<dim; i++) size *=a.size(i);
  float sigma = pow(2.0, -fl);
  float t_min = -pow(2.0, wl-fl-1);
  float t_max = -t_min-sigma;
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  fixed_point_quantize_copy_kernel<<<blockNums, blockSize>>>(a.data<float>(),
                                                           r.data<float>(),
                                                           o.data<float>(),
                                                           size,
                                                           sigma,
                                                           t_min,
                                                           t_max);
  return o;
}

// int main(void){
//   int N = 1<<20;
//   float *a, *r, *o; 
//   cudaMallocManaged(&a, N*sizeof(float));
//   cudaMallocManaged(&r, N*sizeof(float));
//   cudaMallocManaged(&o, N*sizeof(float));

//   for (int i=0; i<N; i++) {
//     a[i] = ((float) rand() / RAND_MAX);
//     r[i] = ((float) rand() / RAND_MAX);
//     o[i] = 0;
//   }

//   int device = -1;
//   cudaGetDevice(&device);
//   cudaMemPrefetchAsync(a, N*sizeof(float), device, NULL);
//   cudaMemPrefetchAsync(r, N*sizeof(float), device, NULL);
//   cudaMemPrefetchAsync(o, N*sizeof(float), device, NULL);

//   // std::cout << "before quantize a: ";
//   // print_array(a, N);
//   // std::cout << "r: ";
//   // print_array(r, N);

//   fixed_point_quantize_copy(a, r, o, N, 3, 2);
//   cudaDeviceSynchronize();
//   fixed_point_quantize_inplace(a, r, N, 3, 2);
//   cudaDeviceSynchronize();

//   // std::cout << "after quantize a: ";
//   // print_array(a,N);
//   // std::cout << "after quantize o: ";
//   // print_array(o,N);

//   cudaFree(a);
//   cudaFree(r);
//   cudaFree(o);
//   return 0;

// }