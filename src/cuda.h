#ifndef CUDA_H
#define CUDA_H

void allocate_memory(short int*& pointer, int height, int width);

void clear_memory(short int*& pointer);

void gaussian(unsigned char* img, float sigma, int rows, int columns, short int*& result);

void cuda_calculate_xy_gradient(short int* img, int height, int width, short int*& grad_x, short int*& grad_y);

#endif