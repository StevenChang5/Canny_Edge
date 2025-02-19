#ifndef UTILS_H
#define UTILS_H

#define PI 3.1415926535
#define EDGE 255
#define NOEDGE 0

void gaussian(unsigned char* img, float sigma, int rows, int columns, short int*& result);

void createGaussianKernel(float*& kernel, float sigma, int* window);

void calculateXYGradient(short int*& img, int rows, int columns, short int*& grad_x, short int*& grad_y);

void sobelOperator(short int*& img, int rows, int columns, short int*& magnitude, short int*& angle);

void nonmaximalSuppression(short int*& grad, short int *& angle, int rows, int columns, short int*& result);

void hysteresis(short int*& edgeCandidates, int rows, int columns, int minVal, int maxVal);

void findEdgePixels(short int*& edgeCandidates, bool*& visited, int start, int minVal, int maxVal, int rows, int columns);

void canny(unsigned char* img, float sigma, int minVal, int maxVal, int height, int width, bool steps);

#endif