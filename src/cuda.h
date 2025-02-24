#ifndef CUDA_H
#define CUDA_H

void cuda_gaussian(unsigned char*& img_h, float sigma, int height, int width, short int*& result_h);

void cuda_sobel(short int*& img_h, int height, int width, short int*& magnitude_h, short int*& angle_h);

void cuda_nonmaixmal_suppression(short int*& magnitude_h, short int*& angle_h, int height, int width, short int*& result_h);

void cuda_canny(unsigned char* img, float sigma, int min_val, int max_val, int height, int width, bool steps);

#endif