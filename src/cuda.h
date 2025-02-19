#ifndef CUDA_H
#define CUDA_H

void allocate_memory(short int*& pointer, int height, int width);

void clear_memory(short int*& pointer);

void cuda_gaussian(unsigned char*& img, float sigma, int rows, int columns, short int*& result);

void cuda_calculate_xy_gradient(short int* img, int height, int width, short int*& grad_x, short int*& grad_y);

void cuda_sobel_operator(short int* grad_x, short int* grad_y, int height, int width, short int*& magnitude, short int*& angle);

void cuda_nonmaixmal_suppression(short int* magnitude, short int* angle, int height, int width, short int*& result);

void cuda_canny(unsigned char* img, float sigma, int minVal, int maxVal, int height, int width, bool steps);

#endif