// cuda.cu
#include "cuda.h"
#include "utils.h"
#include <opencv2/opencv.hpp>

#include <chrono>
#include <iostream>

#define NUM_BLOCKS 1
#define BLOCK_SIZE 32

using namespace cv;
using namespace std;

template <typename T>
struct Image{
    int width;
    int height;
    int stride;
    T* elements;
};

template<typename T>
__device__ T getElement(const Image<T> img, int row, int col){
    return img.elements[row * img.stride + col];
}

template<typename T>
__device__ void setElement(Image<T> img, int row, int col, T value){
    img.elements[row * img.stride + col] = value;
}

template<typename T>
__device__ Image<T> getSubImage(Image<T> img, int row, int col){
    Image<T> sub_img;
    sub_img.width = img.width;
    sub_img.height = img.height;
    sub_img.stride = img.stride;
    sub_img.elements = &img.elements[img.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return sub_img;
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

__global__ void gaussian_util(Image<unsigned char> img_main, float sigma, int window, float* kernel, int height, int width, short int* result){
    /*
        NOTE: Given shared memory, it is not possible to allocate enough space for all sub-images. So, to use tiling
        We need to use a combination of grid-stride technique and tiling. 
    */

    int size = height * width;

    int idx_x = blockIdx.x;
    int idx_y = blockIdx.y;

    int stride_x = blockDim.x * gridDim.x;
    int stride_y = blockDim.y * gridDim.y;

    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;
    
    // Vertical Index (y) = width * (BLOCK_SIZE * blockRow); Horizontal Index (x)) = BLOCK_SIZE * blockCol
    int start_idx = width * (BLOCK_SIZE * blockIdx.y) + (BLOCK_SIZE * blockIdx.x);
    int block_width = BLOCK_SIZE/2;

    __shared__ unsigned char img_s[BLOCK_SIZE][BLOCK_SIZE];

    for(int block_row = blockIdx.y; block_row < height/BLOCK_SIZE; block_row += stride_y){
        for(int block_col = blockIdx.x; block_col < width/BLOCK_SIZE; block_col += stride_x){
            Image<unsigned char> img_sub = getSubImage(img_main, block_row, block_col);
            
            // idx needs to be relative to entire image, not subimage
            int idx_main = ((block_row * BLOCK_SIZE * width) + (thread_row * width) ) + ((block_col * BLOCK_SIZE) + thread_col);
            if(idx_main < size){
                img_s[thread_row][thread_col] = img_main.elements[idx_main];
            }
        }
    }

    __syncthreads();

    __shared__ float img_temp[BLOCK_SIZE][BLOCK_SIZE];

    int center = window/2;

    for(int block_row = blockIdx.y; block_row < height/BLOCK_SIZE; block_row += stride_y){
        for(int block_col = blockIdx.x; block_col < width/BLOCK_SIZE; block_col += stride_x){
            float sum = 0;
            float count = 0;

            for(int k = -center; k < (center+1); k++){
                sum += ((float)(img_main.elements[idx+k]) * kernel[center+k]);
                count += kernel[center+k];
            }
            
            // idx needs to be relative to entire image, not subimage
            int idx_main = ((block_row * BLOCK_SIZE * width) + (thread_row * width) ) + ((block_col * BLOCK_SIZE) + thread_col);
            if(idx_main < size){
                img_s[thread_row][thread_col] = img_main.elements[idx_main];
            }
        }
    }

    float sum = 0;
    float count = 0;
    for(int k = -center; k < (center + 1); k++){
        if(((col+k) >= 0) and ((col+k) < BLOCK_SIZE)){

        }
    }

    
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
  
    for(int )
    
    int center = window/2;

    // Blur in the x direction
    for(int row = idx_y; row < height; row += stride_y){
        for(int col = idx_x; col < width; col += stride_x){
            int idx = row * width + col;
            float sum = 0;
            float count = 0;
            for(int k = -center; k < (center + 1); k++){
                if(((col+k) >= 0) and (col+k) < width){
                    sum += ((float)(img_main.elements[idx+k]) * kernel[center+k]);
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

void cuda_gaussian(unsigned char*& img, float sigma, int height, int width, short int*& result_host){
    int size = height * width;
    int window = 1 + 2 * ceil(3 * sigma);

    Image<unsigned char> img_device;
    img_device.width = img_device.stride = width;
    img_device.height = height;

    short int* result_device;
    result_host = new short int[height*size];

    // Allocate space for img_device.elements and result_device on GPU
    cudaMalloc(&img_device.elements, size*sizeof(unsigned char));
    cudaMalloc(&result_device, size*sizeof(short int));

    // Copy img, kernel to GPU
    cudaMemcpy(img_device.elements, img, size*sizeof(unsigned char), cudaMemcpyHostToDevice);
    float* kernel;
    cudaMallocManaged(&kernel, window * sizeof(float));
    createGaussianKernel(kernel, sigma, window);

    // Perform gaussian blurring
    // dim3 makes the block coordinates of format (x,y), grid of size (BLOCK_SIZE, BLOCK_SIZE)
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(width/dimBlock.x, height/dimBlock.y);
    gaussian_util<<<dimGrid,dimBlock>>>(img_device, sigma, window, kernel, height, width, result_device);

    cudaDeviceSynchronize();

    // Copy blurred image from GPU to CPU
    cudaMemcpy(result_host, result_device, size*sizeof(short int), cudaMemcpyDeviceToHost);

    // Clear elements from GPU memory
    cudaFree(kernel);
    cudaFree(result_device);
    cudaFree(img_device.elements);
}

__global__ void xy_utility(short int* img, int height, int width, short int* grad_x, short int* grad_y){
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
  
    int stride_x = blockDim.x * gridDim.x;
    int stride_y = blockDim.y * gridDim.y;
    
    // Gradient in the x direction, filter in the form of:
    // -1   0   1
    // -2   0   2
    // -1   0   1
    for(int row = idx_y; row < height; row += stride_y){
        for(int col = idx_x; col < width; col += stride_x){
            int pos = row * width + col;
            if(col == 0){
                // Leftmost column, all rows; pad out of bounds with border values
                grad_x[pos] = (2 * img[pos+1]) - (2 * img[pos]);
                // include row above
                if(row != height-1){
                    grad_x[pos] += (img[pos+width+1] - img[pos+width]);
                }
                // include row below
                if(row != 0){
                    grad_x[pos] += (img[pos-width+1] - img[pos-width]);
                }
            }
            else if(col == (width-1)){
                // Rightmost column, all rows; pad out of bounds with border values
                grad_x[pos] = (2 * img[pos]) - (2 * img[pos-1]);
                // include row below
                if(row != height-1){
                    grad_x[pos] += (img[pos+width] - img[pos+width-1]);
                }
                // include row above
                if(row != 0){
                    grad_x[pos] += (img[pos-width] - img[pos-width-1]);
                }
            }
            else{
                grad_x[pos] = (2 * img[pos+1]) - (2 * img[pos-1]);
                if(row != height-1){
                    grad_x[pos] += (img[pos+width+1] - img[pos+width-1]);
                }
                if(row != 0){
                    grad_x[pos] += (img[pos-width+1] - img[pos-width-1]);
                }
            }
        }
    }

    // Gradient in the y direction, filter in the form of:
    //  1   2   1
    //  0   0   0
    // -1  -2  -1
    for(int col = idx_x; col < width; col += stride_x){
        for(int row = idx_y; row < height; row += stride_y){
            int pos = row * width + col;
            if(row == 0){
                // Topmost row, all columns; pad out of bounds with border values
                grad_y[pos] = (2 * img[pos+width]) - (2 * img[pos]);
                if(col != width-1){
                    grad_y[pos] += (img[pos+width+1]-img[pos+1]);
                }
                if(col != 0){
                    grad_y[pos] += (img[pos+width-1]-img[pos-1]);
                }
            }
            else if(row == (height - 1)){
                // Bottommost row, all columns; pad out of bounds with border values
                grad_y[pos] = (2*img[pos]) - (2*img[pos-width]);
                if(col != width-1){
                    grad_y[pos] += (img[pos+1] - img[pos-width+1]);
                }
                if(col != 0){
                    grad_y[pos] += (img[pos-1]-img[pos-width-1]);
                }
            }
            else{
                // Middle, nonborder pixels
                grad_y[pos] = (2*img[pos+width]) - (2*img[pos-width]);
                if(col != width-1){
                    grad_y[pos] += (img[pos+width+1]-img[pos-width+1]);
                }
                if(col != 0){
                    grad_y[pos] += (img[pos+width-1]-img[pos-width-1]);
                }
            }
        }
    }
}

void cuda_calculate_xy_gradient(short int*& img_host, int height, int width, short int*& grad_x_host, short int*& grad_y_host){
    short int* img_device;
    short int* grad_x_device;
    short int* grad_y_device;

    grad_x_host = new short int[height * width];
    grad_y_host = new short int[height * width];

    cudaMalloc(&img_device, height*width*sizeof(short int));
    cudaMemcpy(img_device, img_host, height*width*sizeof(short int), cudaMemcpyHostToDevice);

    cudaMalloc(&grad_x_device, height*width*sizeof(short int));
    cudaMalloc(&grad_y_device, height*width*sizeof(short int));

    xy_utility<<<NUM_BLOCKS,BLOCK_SIZE>>>(img_device, height, width, grad_x_device, grad_y_device);
    
    cudaDeviceSynchronize();

    cudaMemcpy(grad_x_host, grad_x_device, height*width*sizeof(short int), cudaMemcpyDeviceToHost);
    cudaMemcpy(grad_y_host, grad_y_device, height*width*sizeof(short int), cudaMemcpyDeviceToHost);

    cudaFree(grad_x_device);
    cudaFree(grad_y_device);
    cudaFree(img_device);
    delete[] img_host;
}

__global__ void sobel_utility(short int* grad_x, short int* grad_y, int height, int width, short int* magnitude, short int* angle){
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
  
    int stride_x = blockDim.x * gridDim.x;
    int stride_y = blockDim.y * gridDim.y;

    for(int col = idx_x; col < width; col += stride_x){
        for(int row = idx_y; row < height; row += stride_y){
            // Calculate magnitude of gradient at every pixel
            int idx = row * width + col;
            magnitude[idx] = (int)sqrtf((grad_x[idx] * grad_x[idx]) + (grad_y[idx] * grad_y[idx]));

            // Calculate angle of gradient at every pixel
            float temp_angle = atan2((double)grad_y[idx],(double)grad_x[idx]);
            temp_angle *= (180/PI);
            if(temp_angle < 0){
                temp_angle = 360 + temp_angle;
            }
            if((temp_angle >= 22.5 && temp_angle < 67.5) || (temp_angle >= 202.5 && temp_angle < 247.5)){
                angle[idx] = 45;
            }
            else if((temp_angle >= 112.5 && temp_angle < 157.5) || (temp_angle >= 292.5 && temp_angle < 337.5)){
                angle[idx] = 135;
            }
            else if((temp_angle >= 67.5 && temp_angle < 112.5) || (temp_angle >= 247.5 && temp_angle < 292.5)){
                angle[idx] = 90;
            }
            else{
                angle[idx] = 0;
            }
        }
    }
}

void cuda_sobel_operator(short int*& grad_x_host, short int*& grad_y_host, int height, int width, short int*& magnitude_host, short int*& angle_host){
    short int* grad_x_device;
    short int* grad_y_device;
    short int* magnitude_device;
    short int* angle_device;
    magnitude_host = new short int[height*width];
    angle_host = new short int[height*width];
  
    cudaMalloc(&grad_x_device, height*width*sizeof(short int));
    cudaMalloc(&grad_y_device, height*width*sizeof(short int));
    cudaMalloc(&magnitude_device, height*width*sizeof(short int));
    cudaMalloc(&angle_device, height*width*sizeof(short int));

    cudaMemcpy(grad_x_device, grad_x_host, height*width*sizeof(short int), cudaMemcpyHostToDevice);
    cudaMemcpy(grad_y_device, grad_y_host, height*width*sizeof(short int), cudaMemcpyHostToDevice);

    sobel_utility<<<NUM_BLOCKS,BLOCK_SIZE>>>(grad_x_device, grad_y_device, height, width, magnitude_device, angle_device);

    cudaDeviceSynchronize();

    cudaMemcpy(magnitude_host, magnitude_device, height*width*sizeof(short int), cudaMemcpyDeviceToHost);
    cudaMemcpy(angle_host, angle_device, height*width*sizeof(short int), cudaMemcpyDeviceToHost);

    cudaFree(grad_x_device);
    cudaFree(grad_y_device);
    cudaFree(magnitude_device);
    cudaFree(angle_device);
    delete[] grad_x_host;
    delete[] grad_y_host;
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
    short int* grad_x;
    short int* grad_y;
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

    cuda_calculate_xy_gradient(smoothed_img, height, width, grad_x, grad_y);

    
    
    cuda_sobel_operator(grad_x, grad_y, height, width, magnitude, angle);

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
    