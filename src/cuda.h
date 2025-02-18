#ifndef CUDA_H
#define CUDA_H

#define PI 3.1415926535
#define EDGE 255
#define NOEDGE 0

void allocate_memory(short int*& pointer, int height, int width);

void clear_memory(short int*& pointer);

void gaussian(unsigned char* img, float sigma, int rows, int columns, short int*& result);

void cuda_calculate_xy_gradient(short int* img, int height, int width, short int*& grad_x, short int*& grad_y);

void cuda_sobel_operator(short int* grad_x, short int* grad_y, int height, int width, short int*& magnitude, short int*& angle);

void cuda_nonmaixmal_suppression(short int* magnitude, short int* angle, int height, int width, short int*& result);

void cuda_hysteresis(short int*& edges, int height, int width, int min_threshold, int max_threshold);

#endif