#ifndef CONV1D_KERNEL
#define CONV1D_KERNEL

__global__ void conv1DKernel_basic(double *input,
                                   double *output,
                                   int length,
                                   int half_mask_size);

__global__ void conv1DKernel_tiled(double *input,
                                   double *output,
                                   int length,
                                   int half_mask_size);

__global__ void conv1DKernel_simply_tiled(double *input, double *output,
                                          int length, int half_mask_size);

__global__ void conv1DKernel_tiled_dynamic_shared(double *input, double *output, int length,
                                                  int half_mask_size);

void conv1DKernelBasicLauncher(double *input, double *output, double *myMask,
                               int half_mask_size, int N);
void conv1DKernelTiledLauncher(double *input, double *output, double *myMask,
                               int half_mask_size, int N);

void conv1DKernelTiledDynamicSharedLauncher(double *input, double *output, double *myMask,
                                            int half_mask_size, int N);

void conv1DKernelSimplyTiledLauncher(double *input, double *output, double *myMask,
                                     int half_mask_size, int N);

#endif
