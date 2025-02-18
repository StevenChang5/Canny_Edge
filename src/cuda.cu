// cuda.cu
#include "cuda.h"
#include <iostream>

void clear_memory(short int*& pointer){
    cudaFree(pointer);
}

void createGaussianKernel(float*& kernel , float sigma, int window){
    int center = (window)/2;
    float x;
    float sum = 0.0;

    for(int i = 0; i < window; i++){
        x = float(i - center);
        float product = exp(-((x*x)/(2*sigma*sigma)))/(sqrt(6.2831853)*sigma);
        kernel[i] = product;
        sum += product;
    }

    for(int i = 0; i < window; i++){
        kernel[i] /= sum;
    }
}

__global__ void gaussian_util(unsigned char* img, float sigma, int window, int height, int width, float* kernel, float* temp_img, short int* result){
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
  
    int stride_x = blockDim.x * gridDim.x;
    int stride_y = blockDim.y * gridDim.y;
    
    int center = window/2;

    // Blur in the x direction
    for(int row = idx_y; row < height; row += stride_y){
        for(int col = idx_x; col < width; col += stride_x){
            int idx = row * width + col;
            float sum = 0;
            float count = 0;
            for(int k = -center; k < (center + 1); k++){
                if(((col+k) >= 0) and (col+k) < width){
                    sum += ((float)(img[idx+k]) * kernel[center+k]);
                    count += kernel[center+k];
                }
            }
            temp_img[idx] = (sum/count);
        }
    }
    __syncthreads();

    // Blur in the y direction
    for(int col = idx_x; col < width; col += stride_x){
        for(int row = idx_y; row < height; row += stride_y){
            int idx = row * width + col;
            float sum = 0;
            float count = 0;
            for(int k = -center; k < (center + 1); k++){
                if(((row + k) >= 0) and ((row + k) < height)){
                    sum += ((float)(temp_img[idx + (k*width)]) * kernel[center+k]);
                    count += kernel[center+k];
                }
            }
            result[idx] = (short int)(sum/count);
        }
    }
}

void gaussian(unsigned char* img, float sigma, int rows, int columns, short int*& result){
    unsigned char* shared_img;
    float* temp_img; 
    int window = 1 + 2 * ceil(3 * sigma);
    float* kernel;
    
    cudaMallocManaged(&shared_img, rows*columns*sizeof(unsigned char));
    cudaMallocManaged(&temp_img, rows*columns*sizeof(float));
    cudaMallocManaged(&result, rows*columns*sizeof(short int));
    cudaMallocManaged(&kernel, window * sizeof(float));

    createGaussianKernel(kernel, sigma, window);

    for(int i = 0; i < rows*columns; i++){
        shared_img[i] = img[i];
    }

    gaussian_util<<<1,1>>>(shared_img, sigma, window, rows, columns, kernel, temp_img, result);

    cudaDeviceSynchronize();

    cudaFree(shared_img);
    cudaFree(temp_img);
    cudaFree(kernel);
}



