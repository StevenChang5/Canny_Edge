#ifndef UTILS_H
#define UTILS_H

void gaussian(unsigned char* img, float sigma, int rows, int columns, short int*& smoothed_img);

void createGaussianKernel(float*& kernel, float sigma, int* window);

#endif