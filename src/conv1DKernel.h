#ifndef CONV1D_KERNEL
#define CONV1D_KERNEL

void conv1DKernelBasicLauncher(const double *input, double *output, double *myMask,
                               int half_mask_size, int N);
void conv1DKernelTiledLauncher(const double *input, double *output, double *myMask,
                               int half_mask_size, int N);

void conv1DKernelTiledDynamicSharedLauncher(const double *input, double *output, double *myMask,
                                            int half_mask_size, int N);

void conv1DKernelSimplyTiledLauncher(const double *input, double *output, double *myMask,
                                     int half_mask_size, int N);

#endif
