#include <iostream>
#include <cstdlib>
#include <math.h>
#include <cuda.h>

__device__ __forceinline__ float stochastic_round_helper(float a, float r) {
  return floor(a+r);
}

__device__ __forceinline__ float clamp_helper(float a, float min, float max) {
  if (a > max) return max;
  else if (a < min) return min;
  else return a;
}

__device__ __forceinline__ float stochastic_round(float a, float r, int sigma) {
    a /= sigma; 
    a = stochastic_round_helper(a, r);
    a *= sigma;
    return a;
}

__global__ void fixed_point_quantize_cuda(float *a, float *r, int size,
                                     int sigma, int t_min, int t_max) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    a[index] = stochastic_round(a[index], r[index], sigma);
    a[index] = clamp_helper(a[index], t_min, t_max);
  }
}

void fixed_point_quantize(float *a, float *r, int size, int wl, int fl) {
  float sigma = pow(2.0, -fl);
  float t_min = -pow(2.0, wl-fl-1);
  float t_max = -t_min-sigma;
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;
  fixed_point_quantize_cuda<<<blockNums, blockSize>>>(a,
                                                      r,
                                                      size,
                                                      sigma,
                                                      t_min,
                                                      t_max);
}

void print_array(float *a, int size) {
  std::cout << "{";
  for (int i=0; i<size-1; i++) {
    std::cout << a[i] << ",";
  }
  std::cout << a[size-1];
  std::cout << "}" << "\n";
}

int main(void){
  int N = 1<<4;
  float *a, *r; 
  cudaMallocManaged(&a, N*sizeof(float));
  cudaMallocManaged(&r, N*sizeof(float));
  for (int i=0; i<N; i++) {
    a[i] = ((float) rand() / RAND_MAX);
    r[i] = ((float) rand() / RAND_MAX);
  }
  print_array(a, N);
  print_array(r, N);
  fixed_point_quantize(a, r, N, 4, 4);
  cudaDeviceSynchronize();
  print_array(a,N);
  return 0;

}