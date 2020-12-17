#include "cmath"
#include "conv1DKernel.h"
#include <iostream>
#include "time.h"

#define MAX_MASK_SIZE 9

using namespace std;

// THERE IS AN ISSUE WITH THE TILED ALGO
// IT'S NOT ABOUT THE SIZE, SOMETIMES IT FAILS AT SMALL SIZES WHILE SUCCEEDING
// AT LARGER ARRAYS EVEN FOR THE SAME SIZE IT SUCCEEDS AND FAILS RANDOMLY

int main(int argc, char **argv)
{
  //Reading size of the array from execution argv
  size_t p = 10;

  if (argc > 1)
  {
    p = atoi(argv[1]);
  }

  int N{1024};

  N = 1 << p;

  // Setting mask values/weights
  int M = MAX_MASK_SIZE;
  double myMask[M] = {0.001, 0.01, 0.1, 1, 10, 1, 0.1, 0.01, 0.001};

  // Allocating host variables
  double *input, *output, *outputTest, *outputTest2, *outputTest3;

  input = (double *)malloc((int)(N * sizeof(double)));
  output = (double *)malloc((int)(N * sizeof(double)));
  outputTest = (double *)malloc((int)(N * sizeof(double)));
  outputTest2 = (double *)malloc((int)(N * sizeof(double)));
  outputTest3 = (double *)malloc((int)(N * sizeof(double)));

  // Random Input generation
  time_t t;
  srand((unsigned)time(&t));

  for (int i = 0; i < N; i++)
  {
    *(input + i) = rand() % 5 + 1; // 0.1 * (rand() % 5)
  }

  // Launching Conv 1D kernels through their corresponding wrappers
  // M/2 is half the kernel size

  conv1DKernelBasicLauncher(input,
                            output,
                            myMask,
                            M / 2,
                            N);

  conv1DKernelTiledLauncher(input,
                            outputTest,
                            myMask,
                            M / 2,
                            N);

  conv1DKernelSimplyTiledLauncher(input,
                                  outputTest2,
                                  myMask,
                                  M / 2,
                                  N);

  conv1DKernelTiledDynamicSharedLauncher(input,
                                         outputTest3,
                                         myMask,
                                         M / 2,
                                         N);

  //-------------------------------------1-----------------------------------
  cout << "============ Testing Tiled Conv1D ============" << endl;

  int correct = 1;

  for (int i = 0; i < N; i++)
  {
    correct *= int(output[i] == outputTest[i]);
    if (!correct)
    {
      cout << "Left loop, test failed at i = " << i << endl;
      cout << "true value = " << output[i] << endl;
      cout << "false value = " << outputTest[i] << endl;
      break;
    }
  }

  if (correct)
  {
    cout << "============ Test 1 passed successfully ! ============" << endl;
  }

  //-------------------------------------2-----------------------------------

  printf("\n============ Testing Simply Tiled Conv1D ============\n");
  correct = 1;
  for (int i = 0; i < N; i++)
  {
    correct *= int(output[i] == outputTest2[i]);
    if (!correct)
    {
      cout << "Left loop, test failed at i = " << i << endl;
      cout << "true value = " << output[i] << endl;
      cout << "false value = " << outputTest2[i] << endl;
      break;
    }
  }

  if (correct)
  {
    cout << "============ Test 2 passed successfully ! ============" << endl;
  }

  //-------------------------------------3-----------------------------------

  printf("\nTesting Tiled Conv1D with dynamic shared memory\n");
  correct = 1;

  for (int i = 0; i < N; i++)
  {
    correct *= int(output[i] == outputTest3[i]);
    if (!correct)
    {
      cout << "Left loop, test failed at i = " << i << endl;
      cout << "true value = " << output[i] << endl;
      cout << "false value = " << outputTest3[i] << endl;
      break;
    }
  }

  if (correct)
  {
    cout << "============ Test 3 passed successfully ! ============" << endl;
  }

  free(input);
  free(output);
  free(outputTest);
  free(outputTest2);

  return 0;
}