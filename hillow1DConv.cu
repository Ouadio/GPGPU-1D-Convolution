#include "cmath"
#include "conv1DLibrary/conv1DKernel.cuh"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_MASK_SIZE 9

// THERE IS AN ISSUE WITH THE TILED ALGO
// IT'S NOT ABOUT THE SIZE, SOMETIMES IT FAILS AT SMALL SIZES WHILE SUCCEEDING
// AT LARGER ARRAYS EVEN FOR THE SAME SIZE IT SUCCEEDS AND FAILS RANDOMLY
// LOOKS LIKE ITS WAS SOLVED BY MOVING SYNCTHREADS OUTSIDE THE LOOP
// MAYBE IT WAS A RACE CONDITION?
int main(int argc, char **argv) {

  printf("Enter size of your 1D array : ");
  int N{1};
  scanf("%d", &N);

  int M = MAX_MASK_SIZE;
  double myMask[M] = {0.001, 0.01, 0.1, 1, 10, 1, 0.1, 0.01, 0.001};

  double *input, *output, *outputTest, *outputTest2, *outputTest3;

  input = (double *)malloc((int)(N * sizeof(double)));
  output = (double *)malloc((int)(N * sizeof(double)));
  outputTest = (double *)malloc((int)(N * sizeof(double)));
  outputTest2 = (double *)malloc((int)(N * sizeof(double)));
  outputTest3 = (double *)malloc((int)(N * sizeof(double)));

  if (argc > 1) {
    srand(atoi(argv[1]));
  } // Random seed
  else {
    srand(time(0));
  }

  printf("Input data : \n");
  for (int i = 0; i < N; i++) {
    *(input + i) = rand() % 5 + 1; // 0.1 * (rand() % 5)
  }
  for (int i = 0; i < 10; i++) {
    printf("%f\t", input[i]);
  }

  // Launching kernels through their wrappers (Basic then Tiled)

  conv1DKernelBasicLauncher(input, output, myMask, M / 2,
                            N); // M/2 is half the kernel size

  cudaDeviceSynchronize();
  conv1DKernelTiledLauncher(input, outputTest, myMask, M / 2,
                            N); // M/2 is half the kernel size
  cudaDeviceSynchronize();
  conv1DKernelSimplyTiledLauncher(input, outputTest2, myMask, M / 2,
                                  N); // M/2 is half the kernel size
  cudaDeviceSynchronize();

  conv1DKernelTiledDynamicSharedLauncher(input, outputTest3, myMask, M / 2,
                                  N); // M/2 is half the kernel size
  cudaDeviceSynchronize();

  //-------------------------------------1-----------------------------------
  printf("\nTesting Tiled Conv1D\n");
  int correct = 1;

  for (int i = 0; i < N; i++) {
    correct *= (output[i] == outputTest[i] ? 1 : 0);
    if (correct == 0) {
      printf("Left loop, test failed at i = %d \n", i);
      printf("true value = %f \n", output[i]);
      printf("false value = %f \n", outputTest[i]);
      break;
    }
  }

  if (correct == 1) {
    printf("\nTest 1 passed successfully !\n");
  }

  //-------------------------------------2-----------------------------------

  printf("\nTesting Simply Tiled Conv1D\n");
  correct = 1;
  for (int i = 0; i < N; i++) {
    correct *= (output[i] == outputTest2[i] ? 1 : 0);
    if (correct == 0) {
      printf("Left loop, test failed at i = %d \n", i);
      printf("true value = %f \n", output[i]);
      printf("false value = %f \n", outputTest2[i]);
      break;
    }
  }

  if (correct == 1) {
    printf("\nTest 2 passed successfully !\n");
  }

  //-------------------------------------3-----------------------------------

  printf("\nTesting Tiled Conv1D with dynamic shared memory\n");
  correct = 1;

  for (int i = 0; i < N; i++) {
    correct *= (output[i] == outputTest3[i] ? 1 : 0);
    if (correct == 0) {
      printf("Left loop, test failed at i = %d \n", i);
      printf("true value = %f \n", output[i]);
      printf("false value = %f \n", outputTest3[i]);
      break;
    }
  }

  if (correct == 1) {
    printf("\nTest 3 passed successfully !\n");
  }

  free(input);
  free(output);
  free(outputTest);
  free(outputTest2);

  return 0;
}