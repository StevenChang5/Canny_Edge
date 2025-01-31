#include <gtest/gtest.h>
#include <src/utils.h>
#include <opencv2/opencv.hpp>

using namespace cv;

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

TEST(Gaussian, IsNonzero){
  std::string image_path = "/Users/stevenchang/Documents/Repos/Canny_Edge/tests/test.jpg";
  Mat img = cv::imread(image_path, IMREAD_GRAYSCALE);
  unsigned char* data = img.data;
  float sigma = 0.5;
  int rows = 256;
  int columns = 256;
  short int* smoothed_img;

  int sum = 0;

  gaussian(img.data,sigma,rows,columns,smoothed_img);

  for(int i = 0; i < (rows*columns); i++){
    sum += smoothed_img[i];
  }

  EXPECT_NE(sum,0);

  delete[] smoothed_img;
}

TEST(Gaussian, InRange){
  std::string image_path = "/Users/stevenchang/Documents/Repos/Canny_Edge/tests/test.jpg";
  Mat img = imread(image_path, IMREAD_GRAYSCALE);
  unsigned char* data = img.data;
  float sigma = 0.5;
  int rows = 256;
  int columns = 256;
  short int* smoothed_img;

  gaussian(img.data,sigma,rows,columns,smoothed_img);

  for(int i = 0; i < (rows*columns); i++){
    EXPECT_LE(smoothed_img[i], 255);
    EXPECT_GE(smoothed_img[i], 0);
  }

  delete[] smoothed_img;
}

TEST(Gaussian, SameSize){
  std::string image_path = "/Users/stevenchang/Documents/Repos/Canny_Edge/tests/test.jpg";
  Mat img = imread(image_path, IMREAD_GRAYSCALE);
  unsigned char* data = img.data;
  float sigma = 0.5;
  int rows = 256;
  int columns = 256;
  short int* smoothed_img;

  gaussian(img.data,sigma,rows,columns,smoothed_img);

  Mat smoothedMat(rows, columns, CV_16S, smoothed_img);
  EXPECT_EQ(smoothedMat.rows, rows);
  EXPECT_EQ(smoothedMat.cols, columns);
  
  delete[] smoothed_img;
}