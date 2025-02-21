#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <src/cuda.h>

using namespace cv;
using namespace std;

// TEST(CudaGaussian, IsNonzero){
//     string image_path = string(PROJECT_DIR) + "/tests/test.jpg";
//     Mat img = imread(image_path, IMREAD_GRAYSCALE);
//     short int* result;
//     unsigned char* data = img.data;
//     float sigma = 0.5;
//     int rows = 256;
//     int columns = 256;
  
//     int sum = 0;
  
//     cuda_gaussian(data,sigma,rows,columns,result);
  
//     for(int i = 0; i < (rows*columns); i++){
//       cout << result[i] << " ";
//       sum += result[i];
//     }
  
//     EXPECT_NE(sum,0);

//     clear_memory(result);
// }

// TEST(CudaGaussian, InRange){
//     string image_path = string(PROJECT_DIR) + "/tests/test.jpg";
//     Mat img = imread(image_path, IMREAD_GRAYSCALE);
//     short int* result;
//     unsigned char* data = img.data;
//     float sigma = 0.5;
//     int rows = 256;
//     int columns = 256;
  
//     int sum = 0;
  
//     cuda_gaussian(data,sigma,rows,columns,result);
  
//     for(int i = 0; i < (rows*columns); i++){
//       EXPECT_LE(result[i], 255);
//       EXPECT_GE(result[i], 0);
//     }
  
//     clear_memory(result);
// }
  
// TEST(CudaGaussian, GaussianDimensions){
//     string image_path = string(PROJECT_DIR) + "/tests/test.jpg";
//     Mat img = imread(image_path, IMREAD_GRAYSCALE);
//     short int* result;
//     unsigned char* data = img.data;
//     float sigma = 0.5;
//     int rows = 256;
//     int columns = 256;
  
//     int sum = 0;
  
//     cuda_gaussian(data,sigma,rows,columns,result);

//     Mat smoothedMat(rows, columns, CV_16S, result);
//     EXPECT_EQ(smoothedMat.rows, rows);
//     EXPECT_EQ(smoothedMat.cols, columns);

//     clear_memory(result);
// }

// // TEST(CudaGaussian, Visual_Test){
// //     string image_path = string(PROJECT_DIR) + "/tests/test.jpg";
// //     Mat img = imread(image_path, IMREAD_GRAYSCALE);
// //     short int* result;
// //     unsigned char* data = img.data;
// //     float sigma = 0.5;
// //     int rows = 256;
// //     int columns = 256;
  
// //     int sum = 0;
  
// //     gaussian(data,sigma,rows,columns,result);

// //     Mat resultMat(256,256, CV_16S, result);
// //     Mat result_display;

// //     normalize(resultMat, result_display, 0, 255, NORM_MINMAX);
// //     result_display.convertTo(result_display, CV_8U);

// //     imshow("CudaGaussian Visual Test", result_display);
// //     waitKey(0);
// // }

// TEST(CudaGradient, GradientDimensions){
//     short int* grad_x;
//     short int* grad_y;
//     int rows = 3;
//     int columns = 3;
    
//     short int* smoothed_img;
//     allocate_memory(smoothed_img,1,9);

//     short int temp[9]{1,2,1,2,3,2,3,4,3};

//     for(int i = 0; i < 9; i++){
//         smoothed_img[i] = temp[i];
//     }

//     cuda_calculate_xy_gradient(smoothed_img,rows,columns,grad_x,grad_y);
  
//     Mat x(rows, columns, CV_16S, grad_x);
//     EXPECT_EQ(x.rows, rows);
//     EXPECT_EQ(x.cols, columns);
  
//     Mat y(rows, columns, CV_16S, grad_y);
//     EXPECT_EQ(y.rows, rows);
//     EXPECT_EQ(y.cols, columns);
    
//     clear_memory(grad_x);
//     clear_memory(grad_y);
//     clear_memory(smoothed_img);
//   }
  
//   TEST(CudaGradient, xOnes){
//     short int* grad_x;
//     short int* grad_y;
//     int rows = 3;
//     int columns = 3;
    
//     short int* smoothed_img;
//     allocate_memory(smoothed_img,1,9);

//     short int temp[9]{1,1,1,1,1,1,1,1,1};

//     for(int i = 0; i < 9; i++){
//         smoothed_img[i] = temp[i];
//     }
//     cuda_calculate_xy_gradient(smoothed_img,rows,columns,grad_x,grad_y);
  
//     int expected[9]{0,0,0,0,0,0,0,0,0};
  
//     for(int i = 0; i < 9; i++){
//       EXPECT_EQ(grad_x[i],expected[i]);
//       if(grad_x[i] != expected[i]){ cout << "X " << i << endl;}
//     }
  
//     clear_memory(grad_x);
//     clear_memory(grad_y);
//     clear_memory(smoothed_img);
//   }
  
//   TEST(CudaGradient, yOnes){
//     short int* grad_x;
//     short int* grad_y;
//     int rows = 3;
//     int columns = 3;
  
//     short int* smoothed_img;
//     allocate_memory(smoothed_img,1,9);

//     short int temp[9]{1,1,1,1,1,1,1,1,1};

//     for(int i = 0; i < 9; i++){
//         smoothed_img[i] = temp[i];
//     }
//     cuda_calculate_xy_gradient(smoothed_img,rows,columns,grad_x,grad_y);
  
//     int expected[9]{0,0,0,0,0,0,0,0,0};
  
//     for(int i = 0; i < 9; i++){
//       EXPECT_EQ(grad_y[i],expected[i]);
//       if(grad_y[i] != expected[i]){ cout << "Y " << i << endl;}
//     }
  
//     clear_memory(grad_x);
//     clear_memory(grad_y);
//     clear_memory(smoothed_img);
//   }
  
//   TEST(CudaGradient, xCorrect){
//     short int* grad_x;
//     short int* grad_y;
//     int rows = 3;
//     int columns = 3;

//     short int* smoothed_img;
//     allocate_memory(smoothed_img,1,9);

//     short int temp[9]{1,2,1,2,3,2,3,4,3};

//     for(int i = 0; i < 9; i++){
//         smoothed_img[i] = temp[i];
//     }
//     cuda_calculate_xy_gradient(smoothed_img,rows,columns,grad_x,grad_y);
  
//     int expected[9]{3,0,-3,4,0,-4,3,0,-3};
  
//     for(int i = 0; i < 9; i++){
//       EXPECT_EQ(grad_x[i],expected[i]);
//     }
  
//     clear_memory(grad_x);
//     clear_memory(grad_y);
//     clear_memory(smoothed_img);
//   }
  
//   TEST(CudaGradient, yCorrect){
//     short int* grad_x;
//     short int* grad_y;
//     int rows = 3;
//     int columns = 3;

//     short int* smoothed_img;
//     allocate_memory(smoothed_img,1,9);

//     short int temp[9]{1,2,1,2,3,2,3,4,3};

//     for(int i = 0; i < 9; i++){
//         smoothed_img[i] = temp[i];
//     }
//     cuda_calculate_xy_gradient(smoothed_img,rows,columns,grad_x,grad_y);

  
//     int expected[9]{3,4,3,6,8,6,3,4,3};
  
//     for(int i = 0; i < 9; i++){
//       EXPECT_EQ(grad_y[i],expected[i]);
//     }
  
//     clear_memory(grad_x);
//     clear_memory(grad_y);
//     clear_memory(smoothed_img);
//   }

// // TEST(CudaXY, Visual_Test){
// //     string image_path = string(PROJECT_DIR) + "/tests/test.jpg";
// //     Mat img = imread(image_path, IMREAD_GRAYSCALE);
// //     short int* result;
// //     unsigned char* data = img.data;
// //     float sigma = 0.5;
// //     int rows = 256;
// //     int columns = 256;
  
// //     int sum = 0;
  
// //     gaussian(data,sigma,rows,columns,result);

// //     Mat gaussianMat(256,256, CV_16S, result);
// //     Mat gaussian_display;

// //     normalize(gaussianMat, gaussian_display, 0, 255, NORM_MINMAX);
// //     gaussian_display.convertTo(gaussian_display, CV_8U);

// //     imshow("CudaGaussian Visual Test", gaussian_display);
// //     waitKey(0);

// //     short int* grad_x;
// //     short int* grad_y;

// //     cuda_calculate_xy_gradient(result, rows, columns, grad_x, grad_y);
// //     Mat xMat(256,256, CV_16S, grad_x);
// //     Mat yMat(256,256, CV_16S, grad_y);
// //     Mat x_display, y_display;

// //     normalize(xMat, x_display, 0, 255, NORM_MINMAX);
// //     x_display.convertTo(x_display, CV_8U);

// //     normalize(yMat, y_display, 0, 255, NORM_MINMAX);
// //     y_display.convertTo(y_display, CV_8U);

// //     imshow("X Gradient Visual Test", x_display);
// //     waitKey(0);

// //     imshow("Y Gradient Visual Test", y_display);
// //     waitKey(0);

// //     clear_memory(grad_x);
// //     clear_memory(grad_y);
// // }

// TEST(CudaSobelOperator, GradientCalculation){
//     short int* magnitude;
//     short int* angle;
//     int height = 3;
//     int width = 3;

//     short int* grad_x;
//     short int* grad_y;
//     allocate_memory(grad_x,1,9);
//     allocate_memory(grad_y,1,9);

//     short int temp[9]{1,1,1,1,1,1,1,1,1};

//     for(int i = 0; i < 9; i++){
//         grad_x[i] = temp[i];
//         grad_y[i] = temp[i];
//     }

//     short int expectation[9]{1,1,1,1,1,1,1,1,1};

//     cuda_sobel_operator(grad_x, grad_y, height, width, magnitude, angle);

//     for(int i = 0; i < (width * height); i++){
//         EXPECT_EQ(fabs(magnitude[i]-expectation[i]) < FLT_EPSILON, true);
//     }

//     clear_memory(magnitude);
//     clear_memory(angle);
// }

// TEST(CudaSobelOperator, AngleCalculation){
//     short int* magnitude;
//     short int* angle;
//     int height = 1;
//     int width = 5;

//     short int* grad_x;
//     short int* grad_y;
//     allocate_memory(grad_x, height, width);
//     allocate_memory(grad_y, height, width);

//     short int temp_x[5]{1,1,1,1,1};
//     short int temp_y[5]{0,-1,1,3,-3};

//     for(int i = 0; i < 5; i++){
//         grad_x[i] = temp_x[i];
//         grad_y[i] = temp_y[i];
//     }

//     short int expectation[5]{0,135,45,90,90};

//     cuda_sobel_operator(grad_x, grad_y, height, width, magnitude, angle);

//     for(int i = 0; i < (width * height); i++){
//         EXPECT_EQ(angle[i],expectation[i]);
//     }

//     clear_memory(magnitude);
//     clear_memory(angle);
// }

// TEST(CudaNonmaximal, SuppressionCalculation0){
//     short int* result;
//     int height = 3;
//     int width = 3;

//     short int* magnitude;
//     short int* angle;
//     allocate_memory(magnitude, height, width);
//     allocate_memory(angle, height, width);

//     short int temp_magnitude[9]{0,0,0,0,10,0,50,20,50};
//     short int temp_angle[9]{0,0,0,0,0,0,0,0,0};
    
//     for(int i = 0; i < height*width; i++){
//         magnitude[i] = temp_magnitude[i];
//         angle[i] = temp_angle[i];
//     }

//     short int expectation[9]{0,0,0,0,10,0,50,0,50};
  
//     cuda_nonmaixmal_suppression(magnitude,angle,height,width,result);
  
//     for(int i = 0; i < height*width; i++){
//       EXPECT_EQ(result[i],expectation[i]);
//     }
  
//     clear_memory(result);
// }
  
// TEST(CudaNonmaximal, SuppressionCalculation45){
//     short int* result;
//     int height = 3;
//     int width = 3;

//     short int* magnitude;
//     short int* angle;
//     allocate_memory(magnitude, height, width);
//     allocate_memory(angle, height, width);

//     short int temp_magnitude[9]{0,1,1,0,2,0,1,1,0};
//     short int temp_angle[9]{0,45,45,45,45,45,45,45,0};
    
//     for(int i = 0; i < height*width; i++){
//         magnitude[i] = temp_magnitude[i];
//         angle[i] = temp_angle[i];
//     }

//     short int expectation[9]{0,1,0,0,2,0,0,1,0};
  
//     cuda_nonmaixmal_suppression(magnitude,angle,height,width,result);
  
//     for(int i = 0; i < height*width; i++){
//       EXPECT_EQ(result[i],expectation[i]);
//     }
  
//     clear_memory(result);
// }
  
// TEST(CudaNonmaximal, SuppressionCalculation90){
//     short int* result;
//     int height = 3;
//     int width = 3;

//     short int* magnitude;
//     short int* angle;
//     allocate_memory(magnitude, height, width);
//     allocate_memory(angle, height, width);

//     short int temp_magnitude[9]{1,0,0,0,1,0,0,0,1};
//     short int temp_angle[9]{90,90,90,90,90,90,90,90,90};
    
//     for(int i = 0; i < height*width; i++){
//         magnitude[i] = temp_magnitude[i];
//         angle[i] = temp_angle[i];
//     }

//     short int expectation[9]{1,0,0,0,1,0,0,0,1};
  
//     cuda_nonmaixmal_suppression(magnitude,angle,height,width,result);
  
//     for(int i = 0; i < height*width; i++){
//       EXPECT_EQ(result[i],expectation[i]);
//     }
  
//     clear_memory(result);
// }
  
// TEST(CudaNonmaximal, SuppressionCalculation135){
//     short int* result;
//     int height = 3;
//     int width = 3;

//     short int* magnitude;
//     short int* angle;
//     allocate_memory(magnitude, height, width);
//     allocate_memory(angle, height, width);

//     short int temp_magnitude[9]{0,1,1,0,2,0,1,1,0};
//     short int temp_angle[9]{135,135,0,135,135,135,0,135,135};
    
//     for(int i = 0; i < height*width; i++){
//         magnitude[i] = temp_magnitude[i];
//         angle[i] = temp_angle[i];
//     }

//     short int expectation[9]{0,1,0,0,2,0,0,1,0};
  
//     cuda_nonmaixmal_suppression(magnitude,angle,height,width,result);
  
//     for(int i = 0; i < height*width; i++){
//       EXPECT_EQ(result[i],expectation[i]);
//     }
  
//     clear_memory(result);
// }

// TEST(CudaNonmaximal, Visual_Test){
//     string image_path = string(PROJECT_DIR) + "/tests/test.jpg";
//     Mat img = imread(image_path, IMREAD_GRAYSCALE);
//     short int* blurred;
//     unsigned char* data = img.data;
//     float sigma = 0.5;
//     int rows = 256;
//     int columns = 256;
  
//     int sum = 0;
  
//     gaussian(data,sigma,rows,columns,blurred);

//     Mat gaussianMat(256,256, CV_16S, blurred);
//     Mat gaussian_display;

//     normalize(gaussianMat, gaussian_display, 0, 255, NORM_MINMAX);
//     gaussian_display.convertTo(gaussian_display, CV_8U);

//     imshow("CudaGaussian Visual Test", gaussian_display);
//     waitKey(0);

//     short int* grad_x;
//     short int* grad_y;

//     cuda_calculate_xy_gradient(blurred, rows, columns, grad_x, grad_y);
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

//     short int* magnitude;
//     short int* angle;

//     cuda_sobel_operator(grad_x, grad_y, rows, columns, magnitude, angle);

//     short int* result;

//     cuda_nonmaixmal_suppression(magnitude, angle, rows, columns, result);
//     Mat result_mat(256,256, CV_16S, result);
//     Mat result_display;

//     normalize(result_mat, result_display, 0, 255, NORM_MINMAX);
//     result_display.convertTo(result_display, CV_8U);

//     imshow("Nonmaximal Visual Test", result_display);
//     waitKey(0);
// }