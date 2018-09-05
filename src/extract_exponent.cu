#include <iostream>
#include <cstdlib>
#include <climits>
#include <math.h>
#include <cuda.h>


using namespace std;
#define cudaCheckError() {                                          \
  cudaError_t e=cudaGetLastError();                                  \
  if(e!=cudaSuccess) {                                               \
  printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
  exit(0); \
  }                                                                  \
}

#define FULL_MASK (unsigned int) (-1)

__device__ __forceinline__ short extract_exponent(float *a) {
  unsigned int temp = *(reinterpret_cast<unsigned int*>(a));
  temp = (short) (temp << 1 >> 24);
  return temp-(127-1);
}

__inline__ __device__ short warpReduceMax(short val) {
  for (int i=warpSize/2; i > 0; i = i/2) {
    short thread_exponent = __shfl_down_sync(FULL_MASK, val, i);
    val = (val > thread_exponent) ? val : thread_exponent;
  }

  return val;
}

__inline__ __device__ short blockReduceMax(short val) {
  static __shared__ short sdata[32]; 
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warpReduceMax(val);

  if (lane==0) sdata[wid] = val;
  __syncthreads();

  val = (threadIdx.x < blockDim.x / warpSize) ? sdata[lane] : SHRT_MIN;

  if (wid==0) val = warpReduceMax(val);

  return val;
}

__global__ void extract_max_exponent_kernel(float *a, short *o, int size)
{
  short max_exponent = SHRT_MIN;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i=index; i < size; i+= blockDim.x*gridDim.x)  {
    short thread_exponent = extract_exponent(a+i);
    max_exponent = (max_exponent > thread_exponent) ? max_exponent : thread_exponent;
  }
  max_exponent = blockReduceMax(max_exponent);

  if (threadIdx.x == 0) o[blockIdx.x] = max_exponent;

}


__global__ void reduce_max_exponent_kernel(short *a, short *o, int size)
{
  short max_exponent = SHRT_MIN;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i=index; i < size; i+= blockDim.x*gridDim.x)  {
    short thread_exponent = a[i];
    max_exponent = (max_exponent > thread_exponent) ? max_exponent : thread_exponent;
  }
  max_exponent = blockReduceMax(max_exponent);
  if (threadIdx.x == 0) o[blockIdx.x] = max_exponent;
}

void extract_max_exponent(float *a, int size) {
  int blockSize = 1024;
  int blockNums = min((size + blockSize - 1) / blockSize, 1024);

  // extract_max_exponent_kernel<<<blockNums, blockSize>>>(a, o, size); 
  // cudaDeviceSynchronize();
  // cout << "final block reduce\n";
  // for (int i=0; i<20; i++) {
  //   cout << o[i] << "\n";
  // }

  short *out;
  cudaMalloc(&out, blockNums*sizeof(short));
  extract_max_exponent_kernel<<<blockNums, blockSize>>>(a, out, size);
  reduce_max_exponent_kernel<<<1, 1024>>>(out, out, blockNums); 
  cudaFree(out);
}

int main(void)
{
  int N = 128*128*32*32;
  float *x;
  short *max_exp;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&max_exp, min(N, 1024)*sizeof(short));


  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = i;
  }

  int device = -1;
  cudaGetDevice(&device);
  cudaMemPrefetchAsync(x, N*sizeof(float), device, NULL);
  // Run kernel on 1M elements on the GPU
  extract_max_exponent(x, N);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)


  cudaFree(x);
  cudaFree(max_exp);
  
  return 0;
}