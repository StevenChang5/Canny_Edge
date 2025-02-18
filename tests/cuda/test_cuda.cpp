#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <src/cuda.h>

using namespace cv;

TEST(CudaGaussian, IsNonzero){
    std::string image_path = std::string(PROJECT_DIR) + "/tests/test.jpg";
    Mat img = cv::imread(image_path, IMREAD_GRAYSCALE);
    short int* result;
    unsigned char* data = img.data;
    float sigma = 0.5;
    int rows = 256;
    int columns = 256;
  
    int sum = 0;
  
    gaussian(data,sigma,rows,columns,result);
  
    for(int i = 0; i < (rows*columns); i++){
      sum += result[i];
    }
  
    EXPECT_NE(sum,0);

    clear_memory(result);
}

// TEST(CudaGaussian, Works){
//     short int* result;
//     std::string image_path = std::string(PROJECT_DIR) + "/tests/test.jpg";
//     Mat img = cv::imread(image_path, IMREAD_GRAYSCALE);
//     unsigned char* data = img.data;
//     EXPECT_EQ(true, false);

//     gaussian(data, 0.5, 256, 256, result);

//     Mat resultMat(256,256, CV_16S, result);
//     Mat result_display;

//     normalize(resultMat, result_display, 0, 255, NORM_MINMAX);
//     result_display.convertTo(result_display, CV_8U);

//     imshow("Gaussian Smoothed Image", result_display);
//     waitKey(0);
// }