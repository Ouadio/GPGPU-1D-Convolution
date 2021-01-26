#include "cmath"
#include "conv1DKernel.h"
#include <iostream>
#include "time.h"
#include <cuda_runtime.h>
#include <chrono>

#define MAX_MASK_SIZE 33

using namespace std;

// THERE IS AN ISSUE WITH THE TILED ALGO
// IT'S NOT ABOUT THE SIZE, SOMETIMES IT FAILS AT SMALL SIZES WHILE SUCCEEDING
// AT LARGER ARRAYS EVEN FOR THE SAME SIZE IT SUCCEEDS AND FAILS RANDOMLY

int main(int argc, char **argv)
{
  //Reading size of the array from execution argv
  size_t p = 13;

  if (argc > 1)
  {
    p = atoi(argv[1]);
  }

  int N = 1 << p;

  // Setting mask values/weights
  int M = MAX_MASK_SIZE;
  double myMask[M];
  myMask[M / 2] = 10;
  for (size_t i = 1; i <= M / 2; i++)
  {
    myMask[M / 2 + i] = myMask[M / 2 - i] = 10 / (i + 1);
  }

  printf("\nLength of input data = %d and mask size = %d \n", N, M);

  double *input, *outputBase;
  input = (double *)malloc((int)(N * sizeof(double)));
  outputBase = (double *)malloc((int)(N * sizeof(double)));

  // Random Input generation
  time_t t;
  srand((unsigned)time(&t));

  for (int i = 0; i < N; i++)
  {
    *(input + i) = rand() % 10 + 1;
  }

  // Sequential baseline implementation

  auto startSeq = std::chrono::system_clock::now();
  conv1DSequentialLauncher(input,
                           outputBase,
                           myMask,
                           M / 2,
                           N);

  auto endSeq = std::chrono::system_clock::now();

  double timeSeq = std::chrono::duration_cast<std::chrono::milliseconds>(endSeq - startSeq).count();

  cout << "\n=================== Sequential Implementation of Conv1D ====================" << endl;

  cout << "Elapsed time : " << timeSeq << endl;

  // Allocating host variables
  double *outputTest0, *outputTest1, *outputTest2, *outputTest3, *outputTest4;

  // Launching Conv 1D kernels through their corresponding wrappers
  // M/2 is half the kernel size
  float timeBasic, timeTiled, timeTiledStatic, timeTiledDynamic, timeLoop;

  //-------------------------------------0-----------------------------------

  cout << "\n============ Testing Basic GPU Parallel Conv1D ============" << endl;

  int correct = 1;
  outputTest0 = (double *)malloc((int)(N * sizeof(double)));

  conv1DKernelBasicLauncher(input,
                            outputTest0,
                            myMask,
                            M / 2,
                            N,
                            &timeBasic);

  for (int i = 0; i < N; i++)
  {
    correct *= int(outputBase[i] == outputTest0[i]);
    if (!correct)
    {
      cout << "Left loop, test failed at i = " << i << endl;
      cout << "true value = " << outputBase[i] << endl;
      cout << "false value = " << outputTest0[i] << endl;
      break;
    }
  }

  if (correct)
  {
    cout << "============ Test 0 passed successfully ! ============" << endl;
    cout << "Elapsed time : " << timeBasic << endl;
  }

  free(outputTest0);

  //-------------------------------------1-----------------------------------
  cout << "\n============ Testing Tiled Conv1D ============" << endl;

  correct = 1;

  outputTest1 = (double *)malloc((int)(N * sizeof(double)));

  conv1DKernelTiledLauncher(input,
                            outputTest1,
                            myMask,
                            M / 2,
                            N,
                            &timeTiled);

  for (int i = 0; i < N; i++)
  {
    correct *= int(outputBase[i] == outputTest1[i]);
    if (!correct)
    {
      cout << "Left loop, test failed at i = " << i << endl;
      cout << "true value = " << outputBase[i] << endl;
      cout << "false value = " << outputTest1[i] << endl;
      break;
    }
  }

  if (correct)
  {
    cout << "============ Test 1 passed successfully ! ============" << endl;
    cout << "Elapsed time : " << timeTiled << endl;
  }

  free(outputTest1);

  //-------------------------------------2-----------------------------------

  printf("\n============ Testing Simply Tiled Conv1D ============\n");
  correct = 1;

  outputTest2 = (double *)malloc((int)(N * sizeof(double)));

  conv1DKernelSimplyTiledLauncher(input,
                                  outputTest2,
                                  myMask,
                                  M / 2,
                                  N,
                                  &timeTiledStatic);

  for (int i = 0; i < N; i++)
  {
    correct *= int(outputBase[i] == outputTest2[i]);
    if (!correct)
    {
      cout << "Left loop, test failed at i = " << i << endl;
      cout << "true value = " << outputBase[i] << endl;
      cout << "false value = " << outputTest2[i] << endl;
      break;
    }
  }

  if (correct)
  {
    cout << "============ Test 2 passed successfully ! ============" << endl;
    cout << "Elapsed time : " << timeTiledStatic << endl;
  }

  free(outputTest2);

  //-------------------------------------3-----------------------------------

  printf("\n============Testing Tiled Conv1D with dynamic shared memory============\n");
  correct = 1;

  outputTest3 = (double *)malloc((int)(N * sizeof(double)));

  conv1DKernelTiledDynamicSharedLauncher(input,
                                         outputTest3,
                                         myMask,
                                         M / 2,
                                         N,
                                         &timeTiledDynamic);

  for (int i = 0; i < N; i++)
  {
    correct *= int(outputBase[i] == outputTest3[i]);
    if (!correct)
    {
      cout << "Left loop, test failed at i = " << i << endl;
      cout << "true value = " << outputBase[i] << endl;
      cout << "false value = " << outputTest3[i] << endl;
      break;
    }
  }

  if (correct)
  {
    cout << "============ Test 3 passed successfully ! ============" << endl;
    cout << "Elapsed time : " << timeTiledDynamic << endl;
  }

  //-------------------------------------4-----------------------------------

  cout << "\n============ Testing Loop based GPU Parallel Conv1D ============" << endl;

  correct = 1;
  outputTest4 = (double *)malloc((int)(N * sizeof(double)));

  conv1DKernelLoopLauncher(input,
                           outputTest4,
                           myMask,
                           M / 2,
                           N,
                           &timeLoop);

  for (int i = 0; i < N; i++)
  {
    correct *= int(outputBase[i] == outputTest4[i]);
    if (!correct)
    {
      cout << "Left loop, test failed at i = " << i << endl;
      cout << "true value = " << outputBase[i] << endl;
      cout << "false value = " << outputTest4[i] << endl;
      break;
    }
  }

  if (correct)
  {
    cout << "============ Test 4 passed successfully ! ============" << endl;
    cout << "Elapsed time : " << timeLoop << endl;
  }

  free(outputTest4);

  free(input);
  free(outputBase);

  return 0;
}