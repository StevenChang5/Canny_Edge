#include "utils.h"

#include <iostream>
#include <cmath>

using namespace std;

/*
    Pixels of Mat.data are represented as a single array, 
    and access to pixel (row, column) is given by [row * num_columns + column]
*/
void gaussian(unsigned char* img, float sigma, int rows, int columns, short int*& smoothed_img){
    float* kernel;
    int window;

    float* temp;
    createGaussianKernel(kernel,sigma,&window);

    int center = window/2;

    // Create temporary image to store x/y blurring
    float* temp_img = new float[rows*columns];
    // Prepare smoothed image 
    smoothed_img = new short int[rows*columns];

    /* Blur in the x direction */
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < columns; j++){
            float sum = 0;
            float count = 0;
            for(int k = -center; k < (center+1); k++){
                if((j+k >= 0) and (j+k < columns)){
                    sum += (float(img[i * columns + (j+k)]) * kernel[center+k]);
                    count += kernel[center+k];
                }
            }
            temp_img[i * columns + j] = (sum/count);
        }
    }

    /* Blur in the y direction */
    for(int i = 0; i < columns; i++){
        for(int j = 0; j < rows; j++){
            float sum = 0;
            float count = 0; 
            for(int k = -center; k<(center+1); k++){
                if((j+k >= 0) and (j+k < rows)){
                    sum += (float(temp_img[(j+k) * columns + i]) * kernel[center+k]);
                    count += kernel[center+k];
                }
            }
            smoothed_img[j * columns + i] = (short int)(sum/count);
        }   
    }

    delete[] temp_img;
    delete[] kernel;

    return;
    
}

// float*& has same pointer being used outside/inside function, rather than function
// copying address. Necessary for 'kernel = new float[*window]'
void createGaussianKernel(float*& kernel , float sigma, int* window){
    *window = 1 + 2 * ceil(3 * sigma);

    kernel = new float[*window];
    
    int center = (*window)/2;
    float x;
    float sum = 0.0;

    for(int i = 0; i < *window; i++){
        x = float(i - center);
        float product = exp(-((x*x)/(2*sigma*sigma)))/(sqrt(6.2831853)*sigma);
        kernel[i] = product;
        sum += product;
    }

    for(int i = 0; i < *window; i++){
        kernel[i] /= sum;
    }
}