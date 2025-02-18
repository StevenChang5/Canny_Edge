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
      std::cout << result[i] << " ";
      sum += result[i];
    }
  
    EXPECT_NE(sum,0);

    clear_memory(result);
}

TEST(CudaGaussian, InRange){
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
      EXPECT_LE(result[i], 255);
      EXPECT_GE(result[i], 0);
    }
  
    clear_memory(result);
}
  
TEST(CudaGaussian, GaussianDimensions){
    std::string image_path = std::string(PROJECT_DIR) + "/tests/test.jpg";
    Mat img = cv::imread(image_path, IMREAD_GRAYSCALE);
    short int* result;
    unsigned char* data = img.data;
    float sigma = 0.5;
    int rows = 256;
    int columns = 256;
  
    int sum = 0;
  
    gaussian(data,sigma,rows,columns,result);

    Mat smoothedMat(rows, columns, CV_16S, result);
    EXPECT_EQ(smoothedMat.rows, rows);
    EXPECT_EQ(smoothedMat.cols, columns);

    clear_memory(result);
}

// TEST(CudaGaussian, Visual_Test){
//     std::string image_path = std::string(PROJECT_DIR) + "/tests/test.jpg";
//     Mat img = cv::imread(image_path, IMREAD_GRAYSCALE);
//     short int* result;
//     unsigned char* data = img.data;
//     float sigma = 0.5;
//     int rows = 256;
//     int columns = 256;
  
//     int sum = 0;
  
//     gaussian(data,sigma,rows,columns,result);

//     Mat resultMat(256,256, CV_16S, result);
//     Mat result_display;

//     normalize(resultMat, result_display, 0, 255, NORM_MINMAX);
//     result_display.convertTo(result_display, CV_8U);

//     imshow("CudaGaussian Visual Test", result_display);
//     waitKey(0);
// }

TEST(Gradient, GradientDimensions){
    short int* grad_x;
    short int* grad_y;
    int rows = 3;
    int columns = 3;
    
    short int* smoothed_img;
    allocate_memory(smoothed_img,1,9);

    short int temp[9]{1,2,1,2,3,2,3,4,3};

    for(int i = 0; i < 9; i++){
        smoothed_img[i] = temp[i];
    }

    cuda_calculate_xy_gradient(smoothed_img,rows,columns,grad_x,grad_y);
  
    Mat x(rows, columns, CV_16S, grad_x);
    EXPECT_EQ(x.rows, rows);
    EXPECT_EQ(x.cols, columns);
  
    Mat y(rows, columns, CV_16S, grad_y);
    EXPECT_EQ(y.rows, rows);
    EXPECT_EQ(y.cols, columns);
    
    clear_memory(grad_x);
    clear_memory(grad_y);
    clear_memory(smoothed_img);
  }
  
  TEST(Gradient, xOnes){
    short int* grad_x;
    short int* grad_y;
    int rows = 3;
    int columns = 3;
    
    short int* smoothed_img;
    allocate_memory(smoothed_img,1,9);

    short int temp[9]{1,1,1,1,1,1,1,1,1};

    for(int i = 0; i < 9; i++){
        smoothed_img[i] = temp[i];
    }
    cuda_calculate_xy_gradient(smoothed_img,rows,columns,grad_x,grad_y);
  
    int expected[9]{0,0,0,0,0,0,0,0,0};
  
    for(int i = 0; i < 9; i++){
      EXPECT_EQ(grad_x[i],expected[i]);
      if(grad_x[i] != expected[i]){ std::cout << "X " << i << std::endl;}
    }
  
    clear_memory(grad_x);
    clear_memory(grad_y);
    clear_memory(smoothed_img);
  }
  
  TEST(Gradient, yOnes){
    short int* grad_x;
    short int* grad_y;
    int rows = 3;
    int columns = 3;
  
    short int* smoothed_img;
    allocate_memory(smoothed_img,1,9);

    short int temp[9]{1,1,1,1,1,1,1,1,1};

    for(int i = 0; i < 9; i++){
        smoothed_img[i] = temp[i];
    }
    cuda_calculate_xy_gradient(smoothed_img,rows,columns,grad_x,grad_y);
  
    int expected[9]{0,0,0,0,0,0,0,0,0};
  
    for(int i = 0; i < 9; i++){
      EXPECT_EQ(grad_y[i],expected[i]);
      if(grad_y[i] != expected[i]){ std::cout << "Y " << i << std::endl;}
    }
  
    clear_memory(grad_x);
    clear_memory(grad_y);
    clear_memory(smoothed_img);
  }
  
  TEST(Gradient, xCorrect){
    short int* grad_x;
    short int* grad_y;
    int rows = 3;
    int columns = 3;

    short int* smoothed_img;
    allocate_memory(smoothed_img,1,9);

    short int temp[9]{1,2,1,2,3,2,3,4,3};

    for(int i = 0; i < 9; i++){
        smoothed_img[i] = temp[i];
    }
    cuda_calculate_xy_gradient(smoothed_img,rows,columns,grad_x,grad_y);
  
    int expected[9]{3,0,-3,4,0,-4,3,0,-3};
  
    for(int i = 0; i < 9; i++){
      EXPECT_EQ(grad_x[i],expected[i]);
    }
  
    clear_memory(grad_x);
    clear_memory(grad_y);
    clear_memory(smoothed_img);
  }
  
  TEST(Gradient, yCorrect){
    short int* grad_x;
    short int* grad_y;
    int rows = 3;
    int columns = 3;

    short int* smoothed_img;
    allocate_memory(smoothed_img,1,9);

    short int temp[9]{1,2,1,2,3,2,3,4,3};

    for(int i = 0; i < 9; i++){
        smoothed_img[i] = temp[i];
    }
    cuda_calculate_xy_gradient(smoothed_img,rows,columns,grad_x,grad_y);

  
    int expected[9]{3,4,3,6,8,6,3,4,3};
  
    for(int i = 0; i < 9; i++){
      EXPECT_EQ(grad_y[i],expected[i]);
    }
  
    clear_memory(grad_x);
    clear_memory(grad_y);
    clear_memory(smoothed_img);
  }

// TEST(CudaXY, Visual_Test){
//     std::string image_path = std::string(PROJECT_DIR) + "/tests/test.jpg";
//     Mat img = cv::imread(image_path, IMREAD_GRAYSCALE);
//     short int* result;
//     unsigned char* data = img.data;
//     float sigma = 0.5;
//     int rows = 256;
//     int columns = 256;
  
//     int sum = 0;
  
//     gaussian(data,sigma,rows,columns,result);

//     Mat gaussianMat(256,256, CV_16S, result);
//     Mat gaussian_display;

//     normalize(gaussianMat, gaussian_display, 0, 255, NORM_MINMAX);
//     gaussian_display.convertTo(gaussian_display, CV_8U);

//     imshow("CudaGaussian Visual Test", gaussian_display);
//     waitKey(0);

//     short int* grad_x;
//     short int* grad_y;

//     cuda_calculate_xy_gradient(result, rows, columns, grad_x, grad_y);
//     Mat xMat(256,256, CV_16S, grad_x);
//     Mat yMat(256,256, CV_16S, grad_y);
//     Mat x_display, y_display;

//     normalize(xMat, x_display, 0, 255, NORM_MINMAX);
//     x_display.convertTo(x_display, CV_8U);

//     normalize(yMat, y_display, 0, 255, NORM_MINMAX);
//     y_display.convertTo(y_display, CV_8U);

//     imshow("X Gradient Visual Test", x_display);
//     waitKey(0);

//     imshow("Y Gradient Visual Test", y_display);
//     waitKey(0);

//     clear_memory(grad_x);
//     clear_memory(grad_y);
// }