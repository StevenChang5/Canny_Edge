#include <gtest/gtest.h>
#include <src/utils.h>

TEST(Gaussian, KernelCreation) {
  float* kernel = nullptr;
  int window;

  createGaussianKernel(kernel,2,&window);
  for(int i = 0; i < 7; i++){
    EXPECT_EQ(kernel[i],kernel[12-i]);
  }
  EXPECT_EQ(window, 13);

  delete[] kernel;
}