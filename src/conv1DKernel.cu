#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "conv1DKernel.h"
#include "conv1DKernelHeaders.cuh"

#define MAX_MASK_SIZE 9

#define TILE_SIZE 1024

// Dynamic allocation of constant memory is not allowed in CUDA.
__constant__ double myMask_d[MAX_MASK_SIZE];

// Simple 1D Convolution
__global__ void conv1DKernel_basic(const double *input, double *output,
                                   int length, int half_mask_size) {
  int tid = threadIdx.x;
  int i = tid + blockIdx.x * blockDim.x;

  double p{0};
  if (i < length) {

    p = myMask_d[half_mask_size] * input[i];
    for (int j = 1; j < half_mask_size + 1; j++) {
      p += myMask_d[half_mask_size + j] *
           (((i - j < 0) ? 0 : input[i - j]) +
            ((i + j > length) ? 0 : input[i + j]));
    }
    output[i] = p;
  }
}

// Tiled 1D Convolution
__global__ void conv1DKernel_tiled(const double *input, double *output,
                                   int length, int half_mask_size) {
  int tid = threadIdx.x;
  int i = tid + blockIdx.x * blockDim.x;
  int relative_i = i - half_mask_size;

  __shared__ double input_shared[TILE_SIZE + (MAX_MASK_SIZE / 2) * 2];

  if (i < length) {
    input_shared[tid] = relative_i < 0 ? 0 : input[relative_i];
    // Fill the remaining 2*half_mask_size (at the end) using first 2*half_size
    // elements
    if (tid < 2 * half_mask_size) {
      // temp is the size of the input portion within the block
      // it's TILE_SIZE (blockDim.x) except for the 'last' portion of the
      // input array which will be <=TILE_SIZE
      int temp = blockDim.x * (blockIdx.x + 1) < length
                     ? TILE_SIZE
                     : length - blockDim.x * blockIdx.x;

      input_shared[temp + tid] =
          temp + tid + blockIdx.x * blockDim.x - half_mask_size > length - 1
              ? 0
              : input[temp + tid + blockIdx.x * blockDim.x - half_mask_size];
    }
  }
  __syncthreads();

  double p{0};

  if (i < length) {

    p = myMask_d[half_mask_size] * input_shared[tid + half_mask_size];

    for (int j = 1; j < half_mask_size + 1; j++) {
      p += myMask_d[half_mask_size + j] *
           (input_shared[tid + half_mask_size - j] +
            input_shared[tid + half_mask_size + j]);
    }
    output[i] = p;
  }
}

// Simply Tiled 1D Convolution
// The idea is to load only internal cells per TILE in the scratch memory
// Ghost values can be accessed directly from input which is HOPEFULLY still
// IN THE L2 CACHE (not the DRAM ofc)
__global__ void conv1DKernel_simply_tiled(const double *input, double *output,
                                          int length, int half_mask_size) {
  int tid = threadIdx.x;
  int i = tid + blockIdx.x * blockDim.x;

  __shared__ double input_shared[TILE_SIZE];

  if (i < length) {
    input_shared[tid] = input[i];
  }
  __syncthreads();

  double p{0};

  if (i < length) {

    int temp = blockDim.x * (blockIdx.x + 1) < length
                   ? TILE_SIZE
                   : length - blockDim.x * blockIdx.x;

    p = myMask_d[half_mask_size] * input_shared[tid];

    for (int j = 1; j < half_mask_size + 1; j++) {
      if (tid > half_mask_size) {
        if (tid < temp - half_mask_size) {
          p += myMask_d[half_mask_size + j] *
               (input_shared[tid - j] + input_shared[tid + j]);
        } else {
          p += myMask_d[half_mask_size + j] *
               (input_shared[tid - j] + ((i + j) > length ? 0 : input[i + j]));
        }
      } else {
        p += myMask_d[half_mask_size + j] *
             (input_shared[tid + j] + ((i - j) < 0 ? 0 : input[i - j]));
      }
    }
    output[i] = p;
  }
}

// Tiled 1D Convolution with dynamic shared memory
__global__ void conv1DKernel_tiled_dynamic_shared(const double *input,
                                                  double *output, int length,
                                                  int half_mask_size) {
  int tid = threadIdx.x;
  int i = tid + blockIdx.x * blockDim.x;
  int relative_i = i - half_mask_size;

  extern __shared__ double input_shared[];

  if (i < length) {
    input_shared[tid] = relative_i < 0 ? 0 : input[relative_i];
    // Fill the remaining 2*half_mask_size (at the end) using first 2*half_size
    // elements
    if (tid < 2 * half_mask_size) {
      // temp is the size of the input portion within the block
      // it's TILE_SIZE (blockDim.x) except for the 'last' portion of the
      // input array which will be <=TILE_SIZE
      int temp = blockDim.x * (blockIdx.x + 1) < length
                     ? TILE_SIZE
                     : length - blockDim.x * blockIdx.x;

      input_shared[temp + tid] =
          temp + tid + blockIdx.x * blockDim.x - half_mask_size > length - 1
              ? 0
              : input[temp + tid + blockIdx.x * blockDim.x - half_mask_size];
    }
  }
  __syncthreads();

  double p{0};

  if (i < length) {

    p = myMask_d[half_mask_size] * input_shared[tid + half_mask_size];

    for (int j = 1; j < half_mask_size + 1; j++) {
      p += myMask_d[half_mask_size + j] *
           (input_shared[tid + half_mask_size - j] +
            input_shared[tid + half_mask_size + j]);
    }
    output[i] = p;
  }
}

//=======================================================================================
//=======================================================================================
//=================================== WRAPPERS ==========================================
//=======================================================================================
//=======================================================================================

// Wrapper arround basic 1D conv kernel
void conv1DKernelBasicLauncher(const double *input, double *output,
                               double *myMask, int half_mask_size, int N) {

  double *input_d = nullptr;
  double *output_d = nullptr;

  //   cudaError_t cudaStatus;

  cudaMalloc((void **)&input_d, (int)(N * sizeof(double)));
  cudaMalloc((void **)&output_d, (int)(N * sizeof(double)));

  cudaMemcpy(input_d, input, N * sizeof(double), cudaMemcpyHostToDevice);
  // Cte memory copy
  cudaMemcpyToSymbol(myMask_d, myMask, MAX_MASK_SIZE * sizeof(double));

  // Kernel Starts Here

  dim3 blockDim(TILE_SIZE, 1, 1);
  dim3 gridDim(ceil((float)N / TILE_SIZE), 1, 1);

  conv1DKernel_basic<<<gridDim, blockDim>>>(input_d, output_d, N,
                                            half_mask_size);
  // printf("\nBasic kernel used \n");

  // Kernel Ends Here

  cudaMemcpy(output, output_d, N * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(input_d);
  cudaFree(output_d);
}

// Wrapper arround tiled 1D conv kernel
void conv1DKernelTiledLauncher(const double *input, double *output,
                               double *myMask, int half_mask_size, int N) {

  double *input_d = nullptr;
  double *output_d = nullptr;

  //   cudaError_t cudaStatus;

  cudaMalloc((void **)&input_d, (int)(N * sizeof(double)));
  cudaMalloc((void **)&output_d, (int)(N * sizeof(double)));

  cudaMemcpy(input_d, input, N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(myMask_d, myMask, MAX_MASK_SIZE * sizeof(double));

  // Kernel Starts Here

  dim3 blockDim(TILE_SIZE, 1, 1);
  dim3 gridDim(ceil((float)N / TILE_SIZE), 1, 1);

  conv1DKernel_tiled<<<gridDim, blockDim>>>(input_d, output_d, N,
                                            half_mask_size);
  // printf("\nTiled kernel used \n");

  // Kernel Ends Here

  cudaMemcpy(output, output_d, N * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(input_d);
  cudaFree(output_d);
}

// Wrapper arround simplified tiled 1D conv kernel
void conv1DKernelSimplyTiledLauncher(const double *input, double *output,
                                     double *myMask, int half_mask_size,
                                     int N) {

  double *input_d = nullptr;
  double *output_d = nullptr;

  //   cudaError_t cudaStatus;

  cudaMalloc((void **)&input_d, (int)(N * sizeof(double)));
  cudaMalloc((void **)&output_d, (int)(N * sizeof(double)));

  cudaMemcpy(input_d, input, N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(myMask_d, myMask, MAX_MASK_SIZE * sizeof(double));

  // Kernel Starts Here

  dim3 blockDim(TILE_SIZE, 1, 1);
  dim3 gridDim(ceil((float)N / TILE_SIZE), 1, 1);

  conv1DKernel_simply_tiled<<<gridDim, blockDim>>>(input_d, output_d, N,
                                                   half_mask_size);
  // printf("\nSimplified Tiled kernel used \n");

  // Kernel Ends Here

  cudaMemcpy(output, output_d, N * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(input_d);
  cudaFree(output_d);
}

// Wrapper arround tiled 1D conv kernel with dynamic shared memory

void conv1DKernelTiledDynamicSharedLauncher(const double *input, double *output,
                                            double *myMask, int half_mask_size,
                                            int N) {

  double *input_d = nullptr;
  double *output_d = nullptr;

  //   cudaError_t cudaStatus;

  cudaMalloc((void **)&input_d, (int)(N * sizeof(double)));
  cudaMalloc((void **)&output_d, (int)(N * sizeof(double)));

  cudaMemcpy(input_d, input, N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(myMask_d, myMask, MAX_MASK_SIZE * sizeof(double));

  // Kernel Starts Here

  dim3 blockDim(TILE_SIZE, 1, 1);
  dim3 gridDim(ceil((float)N / TILE_SIZE), 1, 1);

  // Here hal_mask_size can be decided at runtime so shared mem allocation is
  // dynamic
  conv1DKernel_tiled_dynamic_shared<<<
      gridDim, blockDim, (TILE_SIZE + 2 * half_mask_size) * sizeof(double)>>>(
      input_d, output_d, N, half_mask_size);
  // printf("\nTiled kernel with Dynamic shared mem used \n");

  // Kernel Ends Here

  cudaMemcpy(output, output_d, N * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(input_d);
  cudaFree(output_d);
}