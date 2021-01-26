#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "conv1DKernel.h"
#include "conv1DKernelHeaders.cuh"

#define MAX_MASK_SIZE 33

#define TILE_SIZE 512

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

    for (int j = 1; j <= half_mask_size; j++) {
      p += myMask_d[half_mask_size + j] *
           (((i - j < 0) ? 0 : input[i - j]) +
            ((i + j >= length) ? 0 : input[i + j]));
    }

    // Number of global mem accesses per thread : 1 + halfMaskSize*2 + 1 +
    // halfMaskSize = 2 + 3*halfMaskSize So global nb of accesses :
    // (2+3*halfMaskSize)*n

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
               (input_shared[tid - j] + ((i + j) >= length ? 0 : input[i + j]));
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

// Nested Loop implementation (More Work per thread) 1D Convolution
__global__ void conv1DKernel_Loop(const double *input, double *output,
                                  int length, int half_mask_size) {
  int tid = threadIdx.x;
  int i = tid + blockIdx.x * blockDim.x;

  double p = 0;

  while (i < length) {

    p = myMask_d[half_mask_size] * input[i];

    for (int j = 1; j <= half_mask_size; j++) {
      p += myMask_d[half_mask_size + j] *
           (((i - j < 0) ? 0 : input[i - j]) +
            ((i + j >= length) ? 0 : input[i + j]));
    }
    output[i] = p;
    i += gridDim.x * blockDim.x;
  }
}

//=======================================================================================
//=======================================================================================
//=================================== WRAPPERS
//==========================================
//=======================================================================================
//=======================================================================================

// Sequential Implementation

void conv1DSequentialLauncher(const double *input, double *output,
                              double *myMask, int half_mask_size, int length) {

  for (size_t i = 0; i < length; i++) {

    output[i] = input[i] * myMask[half_mask_size];

    for (size_t j = 1; j <= half_mask_size; j++) {
      output[i] += ((((int)i - (int)j) < 0 ? 0 : input[i - j]) +
                    ((i + j) >= length ? 0 : input[i + j])) *
                   myMask[half_mask_size + j];
    }
  }
}

// Wrapper arround loop 1D conv kernel
void conv1DKernelLoopLauncher(const double *input, double *output,
                              double *myMask, int half_mask_size, int N,
                              float *time) {

  double *input_d = nullptr;
  double *output_d = nullptr;

  //   cudaError_t cudaStatus;

  cudaMalloc((void **)&input_d, (int)(N * sizeof(double)));
  cudaMalloc((void **)&output_d, (int)(N * sizeof(double)));

  cudaMemcpy(input_d, input, N * sizeof(double), cudaMemcpyHostToDevice);
  // Cte memory copy
  cudaMemcpyToSymbol(myMask_d, myMask, MAX_MASK_SIZE * sizeof(double));

  dim3 blockDim(TILE_SIZE, 1, 1);

  cudaSetDevice(0);
  cudaDeviceProp cudaProp;

  cudaGetDeviceProperties(&cudaProp, 0);
  int nbMultiProcess = cudaProp.multiProcessorCount;

  dim3 gridDim(100 * nbMultiProcess, 1, 1);

  // Computing kernel execution time (memcopy omitted for now)
  cudaEvent_t start, end;

  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start);

  // Kernel Starts Here
  conv1DKernel_Loop<<<gridDim, blockDim>>>(input_d, output_d, N,
                                           half_mask_size);
  printf("Loop kernel Launched :\n");
  printf("Block Dim = (%d, %d, %d) \n", blockDim.x, blockDim.y, blockDim.z);
  printf("Grid Dim = (%d, %d, %d) \n", gridDim.x, gridDim.y, gridDim.z);
  // Kernel Ends Here

  cudaEventRecord(end);

  cudaMemcpy(output, output_d, N * sizeof(double), cudaMemcpyDeviceToHost);

  // Writing elapsed time to (float*) time argument
  cudaEventSynchronize(end);

  cudaEventElapsedTime(time, start, end);

  cudaEventDestroy(start);
  cudaEventDestroy(end);

  cudaFree(input_d);
  cudaFree(output_d);
}

// Wrapper arround basic 1D conv kernel
void conv1DKernelBasicLauncher(const double *input, double *output,
                               double *myMask, int half_mask_size, int N,
                               float *time) {

  double *input_d = nullptr;
  double *output_d = nullptr;

  //   cudaError_t cudaStatus;

  cudaMalloc((void **)&input_d, (int)(N * sizeof(double)));
  cudaMalloc((void **)&output_d, (int)(N * sizeof(double)));

  cudaMemcpy(input_d, input, N * sizeof(double), cudaMemcpyHostToDevice);
  // Cte memory copy
  cudaMemcpyToSymbol(myMask_d, myMask, MAX_MASK_SIZE * sizeof(double));

  dim3 blockDim(TILE_SIZE, 1, 1);
  dim3 gridDim(ceil((float)N / TILE_SIZE), 1, 1);

  // Computing kernel execution time (memcopy omitted for now)
  cudaEvent_t start, end;

  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start);

  // Kernel Starts Here
  conv1DKernel_basic<<<gridDim, blockDim>>>(input_d, output_d, N,
                                            half_mask_size);
  printf("Basic kernel Launched :\n");
  printf("Block Dim = (%d, %d, %d) \n", blockDim.x, blockDim.y, blockDim.z);
  printf("Grid Dim = (%d, %d, %d) \n", gridDim.x, gridDim.y, gridDim.z);
  // Kernel Ends Here

  cudaEventRecord(end);

  cudaMemcpy(output, output_d, N * sizeof(double), cudaMemcpyDeviceToHost);

  // Writing elapsed time to (float*) time argument
  cudaEventSynchronize(end);

  cudaEventElapsedTime(time, start, end);

  cudaEventDestroy(start);
  cudaEventDestroy(end);

  cudaFree(input_d);
  cudaFree(output_d);
}

// Wrapper arround tiled 1D conv kernel
void conv1DKernelTiledLauncher(const double *input, double *output,
                               double *myMask, int half_mask_size, int N,
                               float *time) {

  double *input_d = nullptr;
  double *output_d = nullptr;

  //   cudaError_t cudaStatus;

  cudaMalloc((void **)&input_d, (int)(N * sizeof(double)));
  cudaMalloc((void **)&output_d, (int)(N * sizeof(double)));

  cudaMemcpy(input_d, input, N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(myMask_d, myMask, MAX_MASK_SIZE * sizeof(double));

  dim3 blockDim(TILE_SIZE, 1, 1);
  dim3 gridDim(ceil((float)N / TILE_SIZE), 1, 1);

  // Computing kernel execution time (memcopy omitted for now)
  cudaEvent_t start, end;

  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start);

  // Kernel Starts Here
  conv1DKernel_tiled<<<gridDim, blockDim>>>(input_d, output_d, N,
                                            half_mask_size);
  printf("Tiled kernel Launched :\n");
  printf("Block Dim = (%d, %d, %d) \n", blockDim.x, blockDim.y, blockDim.z);
  printf("Grid Dim = (%d, %d, %d) \n", gridDim.x, gridDim.y, gridDim.z);
  // Kernel Ends Here

  cudaEventRecord(end);

  cudaMemcpy(output, output_d, N * sizeof(double), cudaMemcpyDeviceToHost);

  // Writing elapsed time to (float*) time argument
  cudaEventSynchronize(end);

  cudaEventElapsedTime(time, start, end);

  cudaEventDestroy(start);
  cudaEventDestroy(end);

  cudaFree(input_d);
  cudaFree(output_d);
}

// Wrapper arround simplified tiled 1D conv kernel
void conv1DKernelSimplyTiledLauncher(const double *input, double *output,
                                     double *myMask, int half_mask_size, int N,
                                     float *time) {

  double *input_d = nullptr;
  double *output_d = nullptr;

  //   cudaError_t cudaStatus;

  cudaMalloc((void **)&input_d, (int)(N * sizeof(double)));
  cudaMalloc((void **)&output_d, (int)(N * sizeof(double)));

  cudaMemcpy(input_d, input, N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(myMask_d, myMask, MAX_MASK_SIZE * sizeof(double));

  dim3 blockDim(TILE_SIZE, 1, 1);
  dim3 gridDim(ceil((float)N / TILE_SIZE), 1, 1);

  // Computing kernel execution time (memcopy omitted for now)
  cudaEvent_t start, end;

  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start);

  // Kernel Starts Here
  conv1DKernel_simply_tiled<<<gridDim, blockDim>>>(input_d, output_d, N,
                                                   half_mask_size);
  printf("Simply Tiled kernel Launched :\n");
  printf("Block Dim = (%d, %d, %d) \n", blockDim.x, blockDim.y, blockDim.z);
  printf("Grid Dim = (%d, %d, %d) \n", gridDim.x, gridDim.y, gridDim.z);
  // Kernel Ends Here

  cudaEventRecord(end);

  cudaMemcpy(output, output_d, N * sizeof(double), cudaMemcpyDeviceToHost);

  // Writing elapsed time to (float*) time argument
  cudaEventSynchronize(end);
  cudaEventElapsedTime(time, start, end);

  cudaEventDestroy(start);
  cudaEventDestroy(end);

  cudaFree(input_d);
  cudaFree(output_d);
}

// Wrapper arround tiled 1D conv kernel with dynamic shared memory

void conv1DKernelTiledDynamicSharedLauncher(const double *input, double *output,
                                            double *myMask, int half_mask_size,
                                            int N, float *time) {

  double *input_d = nullptr;
  double *output_d = nullptr;

  //   cudaError_t cudaStatus;

  cudaMalloc((void **)&input_d, (int)(N * sizeof(double)));
  cudaMalloc((void **)&output_d, (int)(N * sizeof(double)));

  cudaMemcpy(input_d, input, N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(myMask_d, myMask, MAX_MASK_SIZE * sizeof(double));

  dim3 blockDim(TILE_SIZE, 1, 1);
  dim3 gridDim(ceil((float)N / TILE_SIZE), 1, 1);

  // Computing kernel execution time (memcopy omitted for now)
  cudaEvent_t start, end;

  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start);

  // Kernel Starts Here
  // Here hal_mask_size can be decided at runtime so shared mem allocation is
  // dynamic
  conv1DKernel_tiled_dynamic_shared<<<
      gridDim, blockDim, (TILE_SIZE + 2 * half_mask_size) * sizeof(double)>>>(
      input_d, output_d, N, half_mask_size);
  printf("Tiled kernel with dynamic Shared memory Launched :\n");
  printf("Block Dim = (%d, %d, %d) \n", blockDim.x, blockDim.y, blockDim.z);
  printf("Grid Dim = (%d, %d, %d) \n", gridDim.x, gridDim.y, gridDim.z);
  // Kernel Ends Here

  cudaEventRecord(end);

  cudaMemcpy(output, output_d, N * sizeof(double), cudaMemcpyDeviceToHost);

  // Writing elapsed time to (float*) time argument
  cudaEventSynchronize(end);
  cudaEventElapsedTime(time, start, end);

  cudaEventDestroy(start);
  cudaEventDestroy(end);

  cudaFree(input_d);
  cudaFree(output_d);
}