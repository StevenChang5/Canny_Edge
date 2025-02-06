#ifndef UTILS_H
#define UTILS_H

#define EDGE 255
#define NOEDGE 0

void gaussian(unsigned char* img, float sigma, int rows, int columns, short int*& smoothed_img);

void createGaussianKernel(float*& kernel, float sigma, int* window);

void calculateXYGradient(short int*& smoothed_img, int rows, int columns, short int*& grad_x, short int*& grad_y);

void approximateGradient(short int*& grad_x, short int*& grad_y, int rows, int columns, short int*& grad);

void approximateAngle(short int*& grad_x, short int*& grad_y, int rows, int columns, short int*& angle);

void nonmaximalSuppression(short int*& grad, short int *& angle, int rows, int columns, short int*& suppress);
#endif