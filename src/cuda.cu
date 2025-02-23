// cuda.cu
#include "cuda.h"
#include "utils.h"
#include <opencv2/opencv.hpp>

#include <chrono>
#include <iostream>

#define NUM_BLOCKS 20
#define BLOCK_SIZE 32

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

void cuda_gaussian(unsigned char*& img, float sigma, int rows, int columns, short int*& result_host){
    unsigned char* img_device;
    float* temp_device; 
    int window = 1 + 2 * ceil(3 * sigma);
    float* kernel;
    short int* result_device;
    result_host = new short int[rows*columns];
    
    cudaMalloc(&img_device, rows*columns*sizeof(unsigned char));
    cudaMalloc(&temp_device, rows*columns*sizeof(float));
    cudaMalloc(&result_device, rows*columns*sizeof(short int));
    cudaMallocManaged(&kernel, window * sizeof(float));

    createGaussianKernel(kernel, sigma, window);

    cudaMemcpy(img_device, img, rows*columns*sizeof(unsigned char), cudaMemcpyHostToDevice);

    gaussian_util<<<NUM_BLOCKS,BLOCK_SIZE>>>(img_device, sigma, window, rows, columns, kernel, temp_device, result_device);

    cudaDeviceSynchronize();

    cudaMemcpy(result_host, result_device, rows*columns*sizeof(short int), cudaMemcpyDeviceToHost);

    cudaFree(img_device);
    cudaFree(temp_device);
    cudaFree(kernel);
    cudaFree(result_device);
}

__global__ void sobel_util(short int* img, int height, int width, short int* magnitude, short int* angle){
    int blk_x = blockIdx.x;
    int blk_y = blockIdx.y;

    int thrd_x = threadIdx.x;
    int thrd_y = threadIdx.y;

    // int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    // int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
  
    int stride_x = gridDim.x;
    int stride_y = gridDim.y;


    __shared__ short int img_shared[BLOCK_SIZE+2][BLOCK_SIZE+2];

    __shared__ short int grad_x[BLOCK_SIZE][BLOCK_SIZE];

    __shared__ short int grad_y[BLOCK_SIZE][BLOCK_SIZE];
    
    for(int row = blk_y; row < height/BLOCK_SIZE; row += stride_y){
        for(int col = blk_x; col < width/BLOCK_SIZE; col += stride_x){
            // Position relative to entire image
            int img_pos = (((row * BLOCK_SIZE * width)+(thrd_y * width)) + ((col * BLOCK_SIZE) + thrd_x));

            img_shared[thrd_y+1][thrd_x+1] = img[img_pos];

            // Fill in outer border
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
        }
    }
}

void cuda_sobel(short int*& img_host, int height, int width, short int*& magnitude_host, short int*& angle_host){
    short int* img_device;
    short int* grad_x_device;
    short int* grad_y_device;
    short int* magnitude_device;
    short int* angle_device;

    magnitude_host = new short int[height*width];
    angle_host = new short int[height*width];

    cudaMalloc(&img_device, height*width*sizeof(short int));
    cudaMemcpy(img_device, img_host, height*width*sizeof(short int), cudaMemcpyHostToDevice);

    cudaMalloc(&grad_x_device, height*width*sizeof(short int));
    cudaMalloc(&grad_y_device, height*width*sizeof(short int));
    cudaMalloc(&magnitude_device, height*width*sizeof(short int));
    cudaMalloc(&angle_device, height*width*sizeof(short int));

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(width/dimBlock.x, height/dimBlock.y);
    sobel_util<<<dimGrid,dimBlock>>>(img_device, height, width, magnitude_device, angle_device);
    
    cudaDeviceSynchronize();

    cudaMemcpy(magnitude_host, magnitude_device, height*width*sizeof(short int), cudaMemcpyDeviceToHost);
    cudaMemcpy(angle_host, angle_device, height*width*sizeof(short int), cudaMemcpyDeviceToHost);

    cudaFree(angle_device);
    cudaFree(magnitude_device);
    cudaFree(grad_x_device);
    cudaFree(grad_y_device);
    cudaFree(img_device);
    delete[] img_host;
}


__global__ void nonmaximal_utility(short int* magnitude, short int* angle, int height, int width, short int* result){
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
  
    int stride_x = blockDim.x * gridDim.x;
    int stride_y = blockDim.y * gridDim.y;

    for(int col = idx_x; col < width; col += stride_x){
        for(int row = idx_y; row < height; row += stride_y){

            int idx = row * width + col;
            bool max = true;

            if(angle[idx] == 0){
                int left = idx - 1;
                int right = idx + 1;
    
                if((idx%width) > 0){
                    if(magnitude[idx] <= magnitude[left]){max = false;}
                }
                if(idx%width < width-1){
                    if(magnitude[idx] <= magnitude[right]){max = false;}
                }
                if(max){result[idx] = magnitude[idx];}
                else{result[idx] = NOEDGE;}
            }   
            else if(angle[idx] == 45){
                int upRight = idx + 1 - width;
                int downLeft = idx - 1 + width;
    
                if((idx%width < width-1) && (idx - width >= 0)){
                    if(magnitude[idx] <= magnitude[upRight]){max = false;}
                }
                if((idx%width > 0) && (idx + width < (height*width))){
                    if(magnitude[idx] <= magnitude[downLeft]){max = false;}
                }
                if(max){result[idx] = magnitude[idx];}
                else{result[idx] = NOEDGE;}
            }
            else if(angle[idx] == 90){
                int up = idx - width;
                int down = idx + width;
    
                if(idx - width >= 0){
                    if(magnitude[idx] <= magnitude[up]){max = false;}
                }
                if(idx + width < (height*width)){
                    if(magnitude[idx] <= magnitude[down]){max = false;}
                }
                if(max){result[idx] = magnitude[idx];}
                else{result[idx] = NOEDGE;}
            } 
            else if(angle[idx] == 135){
                int upLeft = idx - 1 - width;
                int downRight = idx + 1 + width;
    
                if((idx%width > 0) && (idx - width >= 0)){
                    if(magnitude[idx] <= magnitude[upLeft]){max = false;}
                }
                if((idx%width < width-1) && (idx + width < (height*width))){
                    if(magnitude[idx] <= magnitude[downRight]){max = false;}
                }
                if(max){result[idx] = magnitude[idx];}
                else{result[idx] = NOEDGE;}
            }
        }
    }
}

void cuda_nonmaixmal_suppression(short int*& magnitude_host, short int*& angle_host, int height, int width, short int*& result_host){
    short int* magnitude_device;
    short int* angle_device;
    short int* result_device;
    result_host = new short int[height*width];

    cudaMalloc(&magnitude_device, height*width*sizeof(short int));
    cudaMalloc(&angle_device, height*width*sizeof(short int));
    cudaMalloc(&result_device, height*width*sizeof(short int));

    cudaMemcpy(magnitude_device, magnitude_host, height*width*sizeof(short int), cudaMemcpyHostToDevice);
    cudaMemcpy(angle_device, angle_host, height*width*sizeof(short int), cudaMemcpyHostToDevice);

    nonmaximal_utility<<<NUM_BLOCKS,BLOCK_SIZE>>>(magnitude_device,angle_device,height,width,result_device);

    cudaDeviceSynchronize();

    cudaMemcpy(result_host, result_device, height*width*sizeof(short int), cudaMemcpyDeviceToHost);

    cudaFree(result_device);
    cudaFree(magnitude_device);
    cudaFree(angle_device);
    delete[] magnitude_host;
    delete[] angle_host;
}

void cuda_canny(unsigned char* img, float sigma, int minVal, int maxVal, int height, int width, bool steps){
    short int* smoothed_img;    // Image blurred by a Gaussian filter
    short int* magnitude;       // Magnitude of edges, calculated as sqrt(grad_x^2 + grad_y^2)
    short int* angle;           // Angle/direction of edges, calculated as arctan2(grad_y, grad_x)
    short int* nonmaximal;      // Edges w/ nonmaximal suppression applied to neighbors in angle direction

    cuda_gaussian(img,sigma,height,width,smoothed_img);

    if(steps){
        Mat gaussianMat(height,width, CV_16S, smoothed_img);
        Mat gaussian_display;

        normalize(gaussianMat, gaussian_display, 0, 255, NORM_MINMAX);
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

    hysteresis(nonmaximal, height, width, minVal, maxVal);

    // Display final image with canny edge detection applied to it
    Mat finalMat(height,width, CV_16S, nonmaximal);
    Mat final_display;
    normalize(finalMat, final_display, 0, 255, NORM_MINMAX);
    final_display.convertTo(final_display, CV_8U);
    imshow("Final Image", final_display);
    waitKey(0);

    delete[] nonmaximal;
}
    