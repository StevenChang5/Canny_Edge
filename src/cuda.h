#ifndef CUDA_H
#define CUDA_H

void clear_memory(short int*& pointer);

void gaussian(unsigned char* img, float sigma, int rows, int columns, short int*& result);

#endif