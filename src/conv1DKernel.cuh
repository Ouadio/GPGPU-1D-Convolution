#ifndef CONV1D_KERNEL
#define CONV1D_KERNEL

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

void conv1DKernelBasicLauncher(const double *input, double *output, double *myMask,
                               int half_mask_size, int N);
void conv1DKernelTiledLauncher(const double *input, double *output, double *myMask,
                               int half_mask_size, int N);

void conv1DKernelTiledDynamicSharedLauncher(const double *input, double *output, double *myMask,
                                            int half_mask_size, int N);

void conv1DKernelSimplyTiledLauncher(const double *input, double *output, double *myMask,
                                     int half_mask_size, int N);

#endif
