#include "cmath"
#include "conv1DKernel.h"
#include <iostream>

#define MAX_MASK_SIZE 9

using namespace std;

// THERE IS AN ISSUE WITH THE TILED ALGO
// IT'S NOT ABOUT THE SIZE, SOMETIMES IT FAILS AT SMALL SIZES WHILE SUCCEEDING
// AT LARGER ARRAYS EVEN FOR THE SAME SIZE IT SUCCEEDS AND FAILS RANDOMLY
int main(int argc, char **argv)
{

  cout << "Enter size of your 1D array : " << endl;
  int N{1};
  cin >> N;

  int M = MAX_MASK_SIZE;
  double myMask[M] = {0.001, 0.01, 0.1, 1, 10, 1, 0.1, 0.01, 0.001};

  double *input, *output, *outputTest, *outputTest2, *outputTest3;

  input = (double *)malloc((int)(N * sizeof(double)));
  output = (double *)malloc((int)(N * sizeof(double)));
  outputTest = (double *)malloc((int)(N * sizeof(double)));
  outputTest2 = (double *)malloc((int)(N * sizeof(double)));
  outputTest3 = (double *)malloc((int)(N * sizeof(double)));

  cout << "Input data : " << endl;
  for (int i = 0; i < N; i++)
  {
    *(input + i) = rand() % 5 + 1; // 0.1 * (rand() % 5)
  }

  // Launching kernels through their wrappers (Basic then Tiled)

  conv1DKernelBasicLauncher(input, output, myMask, M / 2,
                            N); // M/2 is half the kernel size

  conv1DKernelTiledLauncher(input, outputTest, myMask, M / 2,
                            N); // M/2 is half the kernel size
  conv1DKernelSimplyTiledLauncher(input, outputTest2, myMask, M / 2,
                                  N); // M/2 is half the kernel size

  conv1DKernelTiledDynamicSharedLauncher(input, outputTest3, myMask, M / 2,
                                         N); // M/2 is half the kernel size
  //-------------------------------------1-----------------------------------
  cout << "Testing Tiled Conv1D" << endl;
  int correct = 1;

  for (int i = 0; i < N; i++)
  {
    correct *= (output[i] == outputTest[i] ? 1 : 0);
    if (correct == 0)
    {
      cout << "Left loop, test failed at i = " << i << endl;
      cout << "true value = " << output[i] << endl;
      cout << "false value = " << outputTest[i] << endl;
      break;
    }
  }

  if (correct == 1)
  {
    cout << "Test 1 passed successfully !" << endl;
  }

  //-------------------------------------2-----------------------------------

  printf("\nTesting Simply Tiled Conv1D\n");
  correct = 1;
  for (int i = 0; i < N; i++)
  {
    correct *= (output[i] == outputTest2[i] ? 1 : 0);
    if (correct == 0)
    {
      cout << "Left loop, test failed at i = " << i << endl;
      cout << "true value = " << output[i] << endl;
      cout << "false value = " << outputTest2[i] << endl;
      break;
    }
  }

  if (correct == 1)
  {
    cout << "Test 2 passed successfully !" << endl;
  }

  //-------------------------------------3-----------------------------------

  printf("\nTesting Tiled Conv1D with dynamic shared memory\n");
  correct = 1;

  for (int i = 0; i < N; i++)
  {
    correct *= (output[i] == outputTest3[i] ? 1 : 0);
    if (correct == 0)
    {
      cout << "Left loop, test failed at i = " << i << endl;
      cout << "true value = " << output[i] << endl;
      cout << "false value = " << outputTest3[i] << endl;
      break;
    }
  }

  if (correct == 1)
  {
    cout << "Test 3 passed successfully !" << endl;
  }

  free(input);
  free(output);
  free(outputTest);
  free(outputTest2);

  return 0;
}