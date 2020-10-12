#ifndef CONV1D_KERNEL_HEADER
#define CONV1D_KERNEL_HEADER

__global__ void conv1DKernel_basic(const double *input,
                                   double *output,
                                   int length,
                                   int half_mask_size);

__global__ void conv1DKernel_tiled(const double *input,
                                   double *output,
                                   int length,
                                   int half_mask_size);

__global__ void conv1DKernel_simply_tiled(const double *input, double *output,
                                          int length, int half_mask_size);

__global__ void conv1DKernel_tiled_dynamic_shared(const double *input, double *output, int length,
                                                  int half_mask_size);
#endif