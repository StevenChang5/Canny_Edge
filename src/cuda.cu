// cuda.cu
#include "cuda.h"
#include <iostream>
#include <queue>

using namespace std;

void allocate_memory(short int*& pointer, int height, int width){
    cudaMallocManaged(&pointer, height*width*sizeof(short int));
}

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

    gaussian_util<<<3,512>>>(shared_img, sigma, window, rows, columns, kernel, temp_img, result);

    cudaDeviceSynchronize();

    cudaFree(shared_img);
    cudaFree(temp_img);
    cudaFree(kernel);
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

void cuda_calculate_xy_gradient(short int* img, int height, int width, short int*& grad_x, short int*& grad_y){
    cudaMallocManaged(&grad_x, height*width*sizeof(short int));
    cudaMallocManaged(&grad_y, height*width*sizeof(short int));

    xy_utility<<<3,512>>>(img, height, width, grad_x, grad_y);
    
    cudaDeviceSynchronize();

    cudaFree(img);
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

void cuda_sobel_operator(short int* grad_x, short int* grad_y, int height, int width, short int*& magnitude, short int*& angle){
    cudaMallocManaged(&magnitude, height*width*sizeof(short int));
    cudaMallocManaged(&angle, height*width*sizeof(short int));

    sobel_utility<<<3,512>>>(grad_x, grad_y, height, width, magnitude, angle);

    cudaDeviceSynchronize();

    cudaFree(grad_x);
    cudaFree(grad_y);
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
    
                if((idx%width < width-1) && (idx >= height)){
                    if(magnitude[idx] <= magnitude[upRight]){max = false;}
                }
                if((idx%width > 0) && (idx < (height*width)-height)){
                    if(magnitude[idx] <= magnitude[downLeft]){max = false;}
                }
                if(max){result[idx] = magnitude[idx];}
                else{result[idx] = NOEDGE;}
            }
            else if(angle[idx] == 90){
                int up = idx - width;
                int down = idx + width;
    
                if(idx >= height){
                    if(magnitude[idx] <= magnitude[up]){max = false;}
                }
                if(idx < (height*width)-height){
                    if(magnitude[idx] <= magnitude[down]){max = false;}
                }
                if(max){result[idx] = magnitude[idx];}
                else{result[idx] = NOEDGE;}
            } 
            else if(angle[idx] == 135){
                int upLeft = idx - 1 - width;
                int downRight = idx + 1 + width;
    
                if((idx%width > 0) && (idx >= height)){
                    if(magnitude[idx] <= magnitude[upLeft]){max = false;}
                }
                if((idx%width < width-1) && (idx < (height*width)-height)){
                    if(magnitude[idx] <= magnitude[downRight]){max = false;}
                }
                if(max){result[idx] = magnitude[idx];}
                else{result[idx] = NOEDGE;}
            }
        }
    }
}

void cuda_nonmaixmal_suppression(short int* magnitude, short int* angle, int height, int width, short int*& result){
    cudaMallocManaged(&result, height*width*sizeof(short int));

    nonmaximal_utility<<<3,512>>>(magnitude,angle,height,width,result);

    cudaDeviceSynchronize();

    cudaFree(magnitude);
    cudaFree(angle);
}

__device__ void find_strong_edges(short int* edges, bool*& visited, int start, int min_threshold, int max_threshold, int height, int width){
    edges[start] = EDGE;
    visited[start] = true;
    if(start%width > 0){
        // Check pixel to the bottom left
        if(start < (height*width)-height){
            if(edges[start+width - 1] >= min_threshold && !visited[start+width-1]){
                find_strong_edges(edges, visited, start+width-1, min_threshold, max_threshold, height, width);
                visited[start+width-1] = true;
            }
        }
        // Check pixel to the top left
        if(start >= height){
            if(edges[start-width-1] >= min_threshold && !visited[start-width-1]){
                find_strong_edges(edges, visited, start-width-1, min_threshold, max_threshold, height, width);
                visited[start-width-1] = true;
            }
        }
        // Check pixel to the left
        if(edges[start-1] >= min_threshold && !visited[start-1]){
            find_strong_edges(edges, visited, start-1, min_threshold, max_threshold, height, width);
            visited[start-1] = true;
        }
    }
    if(start%width < width-1){
        // Check pixel to the bottom right
        if(start < (height*width)-height){
            if(edges[start+width+1] >= min_threshold && !visited[start+width+1]){
                find_strong_edges(edges, visited, start+width+1, min_threshold, max_threshold, height, width);
                visited[start+width+1] = true;
            }
        }
        // Check pixel to the top right
        if(start >= height){
            if(edges[start-width+1] >= min_threshold && !visited[start-width+1]){
                find_strong_edges(edges, visited, start-width+1, min_threshold, max_threshold, height, width);
                visited[start-width+1] = true;
            }
        }
        // Check pixel to the right
        if(edges[start+1] >= min_threshold && !visited[start+1]){
            find_strong_edges(edges, visited, start+1, min_threshold, max_threshold, height, width);
            visited[start+1] = true;
        }
    }
    // Check pixel below
    if(start < (height*width)-height){
        if(edges[start+width] >= min_threshold && !visited[start+width]){
            find_strong_edges(edges, visited, start+width, min_threshold, max_threshold, height, width);
            visited[start+width] = true;
        }
    }
    // Check pixel above
    if(start >= height){
        if(edges[start-width] >= min_threshold && !visited[start-width]){
            find_strong_edges(edges, visited, start-width, min_threshold, max_threshold, height, width);
            visited[start-width] = true;
        }
    }


}

__global__ void hysteresis_utility(short int* edges, bool*& visited, int height, int width, int min_threshold, int max_threshold){
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
  
    int stride_x = blockDim.x * gridDim.x;
    int stride_y = blockDim.y * gridDim.y;

    for(int col = idx_x; col < width; col += stride_x){
        for(int row = idx_y; row < height; row += stride_y){
            int index = row * width + col;
            if(edges[index] < min_threshold){
                edges[index] = NOEDGE;
            }
            else if(edges[index] >= max_threshold){
                edges[index] = EDGE;
                // find_strong_edges(edges, visited, index, min_threshold, max_threshold, height, width);
            }
        }
    }
}

void cuda_hysteresis(short int*& edges, int height, int width, int min_threshold, int max_threshold){
    bool* visited;
    cudaMallocManaged(&visited, height*width*sizeof(bool));
    
    hysteresis_utility<<<3,512>>>(edges, visited, height, width, min_threshold, max_threshold);

    cudaFree(visited);
}
