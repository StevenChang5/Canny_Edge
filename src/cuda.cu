// cuda.cu
#include "cuda.h"
#include "utils.h"
#include <opencv2/opencv.hpp>

#include <chrono>
#include <iostream>

#define NUM_BLOCKS 10
#define BLOCK_SIZE 16

using namespace cv;
using namespace std;

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

void cuda_gaussian(unsigned char*& img_h, float sigma, int height, int width, short int*& result_h){
    unsigned char* img_d;
    float* temp_d; 
    int window = 1 + 2 * ceil(3 * sigma);
    float* kernel_g;
    short int* result_d;
    result_h = new short int[height*width];
    
    cudaMalloc(&img_d, height*width*sizeof(unsigned char));
    cudaMalloc(&temp_d, height*width*sizeof(float));
    cudaMalloc(&result_d, height*width*sizeof(short int));
    cudaMallocManaged(&kernel_g, window * sizeof(float));

    createGaussianKernel(kernel_g, sigma, window);

    cudaMemcpy(img_d, img_h, height*width*sizeof(unsigned char), cudaMemcpyHostToDevice);

    gaussian_util<<<NUM_BLOCKS,BLOCK_SIZE>>>(img_d, sigma, window, height, width, kernel_g, temp_d, result_d);

    cudaDeviceSynchronize();

    cudaMemcpy(result_h, result_d, height*width*sizeof(short int), cudaMemcpyDeviceToHost);

    cudaFree(img_d);
    cudaFree(temp_d);
    cudaFree(result_d);
    cudaFree(kernel_g);
}

__global__ void sobel_util(short int* img, int height, int width, short int* magnitude, short int* angle){
    int blk_x = blockIdx.x;
    int blk_y = blockIdx.y;

    int thrd_x = threadIdx.x;
    int thrd_y = threadIdx.y;
  
    int stride_x = gridDim.x;
    int stride_y = gridDim.y;


    __shared__ short int img_shared[BLOCK_SIZE+2][BLOCK_SIZE+2];

    __shared__ short int grad_x[BLOCK_SIZE][BLOCK_SIZE];

    __shared__ short int grad_y[BLOCK_SIZE][BLOCK_SIZE];
    
    for(int row = blk_y; row < height/BLOCK_SIZE; row += stride_y){
        for(int col = blk_x; col < width/BLOCK_SIZE; col += stride_x){
            /********************************************************************************** 
            * Tiling, copy the thread block's designated pixels to the GPU
            **********************************************************************************/
            int img_pos = (((row * BLOCK_SIZE * width)+(thrd_y * width)) + ((col * BLOCK_SIZE) + thrd_x));

            img_shared[thrd_y+1][thrd_x+1] = img[img_pos];

            // Fill in outer border for kernel calculation on edge pixels
            // left/right border
            if(blk_x == 0){
                img_shared[thrd_y+1][thrd_x] = img[img_pos];
            }else{
                img_shared[thrd_y+1][thrd_x] = img[img_pos - 1];
            }
            if(blk_x == ((width/BLOCK_SIZE)-1)){
                img_shared[thrd_y+1][thrd_x+2] = -1;
            }else{
                img_shared[thrd_y+1][thrd_x+2] = img[img_pos+1];
            }
            
            // top/bottom border
            if(blk_y == 0){
                img_shared[thrd_y][thrd_x+1] = img[img_pos];
            }else{
                img_shared[thrd_y][thrd_x+1] = img[img_pos - width];
            }
            if(blk_y == ((height/BLOCK_SIZE)-1)){
                img_shared[thrd_y+2][thrd_x+1] = -1;
            }else{
                img_shared[thrd_y+2][thrd_x+1] = img[img_pos + width];
            }
            
            // corners
            if(blk_x == 0 and blk_y == 0){
                img_shared[thrd_y][thrd_x] = img[img_pos];
            }else{
                img_shared[thrd_y][thrd_x] = img[img_pos - 1 - width];
            }

            if(blk_x == 0 and blk_y == ((height/BLOCK_SIZE)-1)){
                img_shared[thrd_y+2][thrd_x] = img[img_pos];
            }else{
                img_shared[thrd_y+2][thrd_x] = img[img_pos -1 + width];
            }

            if(blk_x == ((width/BLOCK_SIZE)-1) and blk_y == 0){
                img_shared[thrd_y][thrd_x+2] = img[img_pos];
            }else{
                img_shared[thrd_y][thrd_x+2] = img[img_pos +1 - width];
            }

            if(blk_x == ((width/BLOCK_SIZE)-1) and blk_y == ((height/BLOCK_SIZE)-1)){
                img_shared[thrd_y+2][thrd_x+2] = img[img_pos];
            }else{
                img_shared[thrd_y+2][thrd_x+2] = img[img_pos +1 + width];
            }

            __syncthreads();

            /********************************************************************************** 
            * Sobel filter calculation
            **********************************************************************************/
            grad_x[thrd_y][thrd_x] = (2 * img_shared[thrd_y+1][thrd_x+2]) - (2 * img_shared[thrd_y+1][thrd_x]);
            grad_x[thrd_y][thrd_x] += img_shared[thrd_y+2][thrd_x+2] - img_shared[thrd_y+2][thrd_x];
            grad_x[thrd_y][thrd_x] += img_shared[thrd_y][thrd_x+2] - img_shared[thrd_y][thrd_x];

            grad_y[thrd_y][thrd_x] = (2 * img_shared[thrd_y][thrd_x+1]) - (2 * img_shared[thrd_y+2][thrd_x+1]);
            grad_y[thrd_y][thrd_x] += img_shared[thrd_y][thrd_x+2] - img_shared[thrd_y+2][thrd_x+2];
            grad_y[thrd_y][thrd_x] += img_shared[thrd_y][thrd_x] - img_shared[thrd_y+2][thrd_x];

            magnitude[img_pos] = (short int)sqrtf((grad_x[thrd_y][thrd_x] * grad_x[thrd_y][thrd_x]) 
                                                + (grad_y[thrd_y][thrd_x] * grad_y[thrd_y][thrd_x]));

            // Calculate angle of gradient at every pixel
            float temp_angle = atan2((double)grad_y[thrd_y][thrd_x],(double)grad_x[thrd_y][thrd_x]);
            temp_angle *= (180/PI);
            if(temp_angle < 0){
                temp_angle = 360 + temp_angle;
            }
            if((temp_angle >= 22.5 && temp_angle < 67.5) || (temp_angle >= 202.5 && temp_angle < 247.5)){
                angle[img_pos] = 45;
            }
            else if((temp_angle >= 112.5 && temp_angle < 157.5) || (temp_angle >= 292.5 && temp_angle < 337.5)){
                angle[img_pos] = 135;
            }
            else if((temp_angle >= 67.5 && temp_angle < 112.5) || (temp_angle >= 247.5 && temp_angle < 292.5)){
                angle[img_pos] = 90;
            }
            else{
                angle[img_pos] = 0;
            }

            __syncthreads();
        }
    }
}

void cuda_sobel(short int*& img_h, int height, int width, short int*& magnitude_h, short int*& angle_h){
    short int* img_d;
    short int* magnitude_d;
    short int* angle_d;

    magnitude_h = new short int[height*width];
    angle_h = new short int[height*width];

    cudaMalloc(&img_d, height*width*sizeof(short int));
    cudaMemcpy(img_d, img_h, height*width*sizeof(short int), cudaMemcpyHostToDevice);

    cudaMalloc(&magnitude_d, height*width*sizeof(short int));
    cudaMalloc(&angle_d, height*width*sizeof(short int));

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(width/dimBlock.x, height/dimBlock.y);
    sobel_util<<<dimGrid,dimBlock>>>(img_d, height, width, magnitude_d, angle_d);
    
    cudaDeviceSynchronize();

    cudaMemcpy(magnitude_h, magnitude_d, height*width*sizeof(short int), cudaMemcpyDeviceToHost);
    cudaMemcpy(angle_h, angle_d, height*width*sizeof(short int), cudaMemcpyDeviceToHost);

    cudaFree(angle_d);
    cudaFree(magnitude_d);
    cudaFree(img_d);
}


__global__ void nonmaximal_utility(short int* magnitude, short int* angle, int height, int width, short int* result){
    int blk_x = blockIdx.x;
    int blk_y = blockIdx.y;

    int thrd_x = threadIdx.x;
    int thrd_y = threadIdx.y;
  
    int stride_x = gridDim.x;
    int stride_y = gridDim.y;


    __shared__ short int img_shared[BLOCK_SIZE+2][BLOCK_SIZE+2];

    for(int row = blk_y; row < height/BLOCK_SIZE; row += stride_y){
        for(int col = blk_x; col < width/BLOCK_SIZE; col += stride_x){
            /********************************************************************************** 
            * Tiling, copy the thread block's designated pixels to the GPU
            **********************************************************************************/
            // Position relative to entire image
            int img_pos = (((row * BLOCK_SIZE * width)+(thrd_y * width)) + ((col * BLOCK_SIZE) + thrd_x));

            // Fill in outer border
            // left/right border
            if(blk_x == 0){
                img_shared[thrd_y+1][thrd_x] = magnitude[img_pos];
            }else{
                img_shared[thrd_y+1][thrd_x] = magnitude[img_pos - 1];
            }
            if(blk_x == ((width/BLOCK_SIZE)-1)){
                img_shared[thrd_y+1][thrd_x+2] = magnitude[img_pos];
            }else{
                img_shared[thrd_y+1][thrd_x+2] = magnitude[img_pos+1];
            }
            
            // top/bottom border
            if(blk_y == 0){
                img_shared[thrd_y][thrd_x+1] = magnitude[img_pos];
            }else{
                img_shared[thrd_y][thrd_x+1] = magnitude[img_pos - width];
            }
            if(blk_y == ((height/BLOCK_SIZE)-1)){
                img_shared[thrd_y+2][thrd_x+1] = magnitude[img_pos];
            }else{
                img_shared[thrd_y+2][thrd_x+1] = magnitude[img_pos + width];
            }
            
            // corners
            if(blk_x == 0 and blk_y == 0){
                img_shared[thrd_y][thrd_x] = magnitude[img_pos];
            }else{
                img_shared[thrd_y][thrd_x] = magnitude[img_pos - 1 - width];
            }

            if(blk_x == 0 and blk_y == ((height/BLOCK_SIZE)-1)){
                img_shared[thrd_y+2][thrd_x] = magnitude[img_pos];
            }else{
                img_shared[thrd_y+2][thrd_x] = magnitude[img_pos -1 + width];
            }

            if(blk_x == ((width/BLOCK_SIZE)-1) and blk_y == 0){
                img_shared[thrd_y][thrd_x+2] = magnitude[img_pos];
            }else{
                img_shared[thrd_y][thrd_x+2] = magnitude[img_pos +1 - width];
            }

            if(blk_x == ((width/BLOCK_SIZE)-1) and blk_y == ((height/BLOCK_SIZE)-1)){
                img_shared[thrd_y+2][thrd_x+2] = magnitude[img_pos];
            }else{
                img_shared[thrd_y+2][thrd_x+2] = magnitude[img_pos +1 + width];
            }

            __syncthreads();

            /********************************************************************************** 
            * Perform nonmaximal suppression on pixels
            **********************************************************************************/
            bool max = true;

            if(angle[img_pos] == 0){
                if(img_shared[thrd_y+1][thrd_x+1] <= img_shared[thrd_y+1][thrd_x]){
                    max = false;}
                if(img_shared[thrd_y+1][thrd_x+1] <= img_shared[thrd_y+1][thrd_x+2]){
                    max = false;}
                if(max){result[img_pos] = img_shared[thrd_y+1][thrd_x+1];}
                else{result[img_pos] = NOEDGE;}
            }   
            else if(angle[img_pos] == 45){
                if(img_shared[thrd_y+1][thrd_x+1] <= img_shared[thrd_y][thrd_x+2]){
                    max = false;}
                if(img_shared[thrd_y+1][thrd_x+1] <= img_shared[thrd_y+2][thrd_x]){
                    max = false;}
                if(max){result[img_pos] = img_shared[thrd_y+1][thrd_x+1];}
                else{result[img_pos] = NOEDGE;}
            }
            else if(angle[img_pos] == 90){
                if(img_shared[thrd_y+1][thrd_x+1] <= img_shared[thrd_y][thrd_x+1]){
                    max = false;}
                if(img_shared[thrd_y+1][thrd_x+1] <= img_shared[thrd_y+2][thrd_x+1]){
                    max = false;}
                if(max){result[img_pos] = img_shared[thrd_y+1][thrd_x+1];}
                else{result[img_pos] = NOEDGE;}
            } 
            else if(angle[img_pos] == 135){
                if(img_shared[thrd_y+1][thrd_x+1] <= img_shared[thrd_y][thrd_x]){
                    max = false;}
                if(img_shared[thrd_y+1][thrd_x+1] <= img_shared[thrd_y+2][thrd_x+2]){
                    max = false;}
                if(max){result[img_pos] = img_shared[thrd_y+1][thrd_x+1];}
                else{result[img_pos] = NOEDGE;}
            }

            __syncthreads();
        }
    }
}


void cuda_nonmaixmal_suppression(short int*& magnitude_h, short int*& angle_h, int height, int width, short int*& result_h){
    short int* magnitude_d;
    short int* angle_d;
    short int* result_d;
    result_h = new short int[height*width];

    cudaMalloc(&magnitude_d, height*width*sizeof(short int));
    cudaMalloc(&angle_d, height*width*sizeof(short int));
    cudaMalloc(&result_d, height*width*sizeof(short int));

    cudaMemcpy(magnitude_d, magnitude_h, height*width*sizeof(short int), cudaMemcpyHostToDevice);
    cudaMemcpy(angle_d, angle_h, height*width*sizeof(short int), cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(width/dimBlock.x, height/dimBlock.y);
    nonmaximal_utility<<<dimGrid,dimBlock>>>(magnitude_d,angle_d,height,width,result_d);

    cudaDeviceSynchronize();

    cudaMemcpy(result_h, result_d, height*width*sizeof(short int), cudaMemcpyDeviceToHost);

    cudaFree(result_d);
    cudaFree(magnitude_d);
    cudaFree(angle_d);
}

void cuda_canny(unsigned char* img, float sigma, int min_val, int max_val, int height, int width, bool steps){
    short int* smoothed_img;    // Image blurred by a Gaussian filter
    short int* magnitude;       // Magnitude of edges, calculated as sqrt(grad_x^2 + grad_y^2)
    short int* angle;           // Angle/direction of edges, calculated as arctan2(grad_y, grad_x)
    short int* nonmaximal;      // Edges w/ nonmaximal suppression applied to neighbors in angle direction

    cuda_gaussian(img,sigma,height,width,smoothed_img);

    if(steps){
        Mat gaussian_mat(height,width, CV_16S, smoothed_img);
        Mat gaussian_display;

        normalize(gaussian_mat, gaussian_display, 0, 255, NORM_MINMAX);
        gaussian_display.convertTo(gaussian_display, CV_8U);

        imshow("CudaGaussian Visual Test", gaussian_display);
        waitKey(0);
    }

    cuda_sobel(smoothed_img, height, width, magnitude, angle);

    if(steps){
        Mat sobel_mat(height,width, CV_16S, magnitude);
        Mat sobel_display;

        normalize(sobel_mat, sobel_display, 0, 255, NORM_MINMAX);
        sobel_display.convertTo(sobel_display, CV_8U);

        imshow("Sobel Visual Test", sobel_display);
        waitKey(0);
    }
    cuda_nonmaixmal_suppression(magnitude, angle, height, width, nonmaximal);

    if(steps){
        Mat result_mat(height,width, CV_16S, nonmaximal);
        Mat result_display;

        normalize(result_mat, result_display, 0, 255, NORM_MINMAX);
        result_display.convertTo(result_display, CV_8U);

        imshow("Nonmaximal Visual Test", result_display);
        waitKey(0);
    }

    hysteresis(nonmaximal, height, width, min_val, max_val);

    // Display final image with canny edge detection applied to it
    Mat finalMat(height,width, CV_16S, nonmaximal);
    Mat final_display;
    normalize(finalMat, final_display, 0, 255, NORM_MINMAX);
    final_display.convertTo(final_display, CV_8U);
    imshow("Final Image", final_display);
    waitKey(0);

    delete[] smoothed_img;
    delete[] nonmaximal;
    delete[] magnitude;
    delete[] angle;
}
    