#include <gtest/gtest.h>
#include <src/utils.h>

TEST(Gaussian, KernelSumOne){
  float* kernel;
  int window;
  float sum = 0;

  createGaussianKernel(kernel,0.5,&window);
  for(int i = 0; i < window; i++){
    sum += kernel[i];
  }
  EXPECT_EQ(fabs(sum-1) < FLT_EPSILON, true);

  delete[] kernel;
}

TEST(Gaussian, KernelValues){
  float* kernel;
  int window;
  float expected[5]{0.0002638651, 0.1064507720, 0.7865707259, 0.1064507720, 0.0002638651};

  createGaussianKernel(kernel,0.5,&window);
  for(int i = 0; i < window; i++){
    EXPECT_EQ(fabs(expected[i]-kernel[i]) < FLT_EPSILON,true);
  }

  delete[] kernel;
}

TEST(Gaussian, KernelCreation) {
  float* kernel;
  int window;

  createGaussianKernel(kernel,2,&window);
  for(int i = 0; i < 7; i++){
    EXPECT_EQ(kernel[i],kernel[12-i]);
  }
  EXPECT_EQ(window, 13);

  delete[] kernel;
}

TEST(Gaussian, GaussianBlur){
  unsigned char* img = new unsigned char[25];
  for(int i = 0; i < 25; i++){
    img[i] = '20';
  }
  float sigma = 0.5;
  int rows = 5;
  int columns = 5;
  short int* smoothed_img;
  gaussian(img,sigma,rows,columns,smoothed_img);
  for(int i = 0; i < 2; i++){
    EXPECT_EQ(smoothed_img[i], 2);
  }

  delete[] img;
}