#ifndef UTILS_H
#define UTILS_H

void gaussian(unsigned char* img, int size);

void createGaussianKernel(float*& kernel, float sigma, int* window);

#endif