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

// Calculate gradient in the x and y direction
// Access to pixel (row, column) is given by [row * num_columns + column]
void calculateXYGradient(short int*& smoothed_img, int rows, int columns, short int*& grad_x, short int*& grad_y){
    grad_x = new short int[rows * columns];
    grad_y = new short int[rows * columns];

    // x derivative
    for(int r = 0; r < rows; r++){
        int pos = r * columns;
        // Leftmost column, all rows; pad out of bounds with border values
        grad_x[pos] = (2 * smoothed_img[pos+1]) - (2 * smoothed_img[pos]);
        // include row above
        if(r != rows-1){
            grad_x[pos] += (smoothed_img[pos+columns+1] - smoothed_img[pos+columns]);
        }
        // include row below
        if(r != 0){
            grad_x[pos] += (smoothed_img[pos-columns+1] - smoothed_img[pos-columns]);
        }
        // pos and c are both for tracking column, but one tracks actual value in array and other tracks high-level value
        pos++;

        // Middle, non-border pixels
        for(int c = 1; c < columns-1; c++, pos++){
            grad_x[pos] = (2 * smoothed_img[pos+1]) - (2 * smoothed_img[pos-1]);
            if(r != rows-1){
                grad_x[pos] += (smoothed_img[pos+columns+1] - smoothed_img[pos+columns-1]);
            }
            if(r != 0){
                grad_x[pos] += (smoothed_img[pos-columns+1] - smoothed_img[pos-columns-1]);
            }
        }

        // Rightmost column, all rows; pad out of bounds with border values
        grad_x[pos] = (2 * smoothed_img[pos]) - (2 * smoothed_img[pos-1]);
        // include row below
        if(r != rows-1){
            grad_x[pos] += (smoothed_img[pos+columns] - smoothed_img[pos+columns-1]);
        }
        // include row above
        if(r != 0){
            grad_x[pos] += (smoothed_img[pos-columns] - smoothed_img[pos-columns-1]);
        }
    }

    // y derivative
    // Access to pixel (row, column) is given by [row * num_columns + column]
    for(int c = 0; c < columns; c++){
        int pos = c;
        // topmost row, all columns; pad out of bounds with border values
        grad_y[pos] = (2 * smoothed_img[pos+columns]) - (2 * smoothed_img[pos]);
        if(c != columns-1){
            grad_y[pos] += (smoothed_img[pos+columns+1]-smoothed_img[pos+1]);
        }
        if(c != 0){
            grad_y[pos] += (smoothed_img[pos+columns-1]-smoothed_img[pos-1]);
        }
        pos += columns;

        // Middle, nonborder pixels
        for(int r = 1; r < rows-1; r++, pos+=columns){
            grad_y[pos] = (2*smoothed_img[pos+columns]) - (2*smoothed_img[pos-columns]);
            if(c != columns-1){
                grad_y[pos] += (smoothed_img[pos+columns+1]-smoothed_img[pos-columns+1]);
            }
            if(c != 0){
                grad_y[pos] += (smoothed_img[pos+columns-1]-smoothed_img[pos-columns-1]);
            }
        }

        // bottommost row, all columns; pad out of bounds with border values
        grad_y[pos] = (2*smoothed_img[pos]) - (2*smoothed_img[pos-columns]);
        if(c != columns-1){
            grad_y[pos] += (smoothed_img[pos+1] - smoothed_img[pos-columns+1]);
        }
        if(c != 0){
            grad_y[pos] += (smoothed_img[pos-1]-smoothed_img[pos-columns-1]);
        }
    }
}

void approximateGradient(short int*& grad_x, short int*& grad_y, int rows, int columns, short int*& grad){
    grad = new short int[rows * columns];
    for(int i = 0; i < (rows*columns); i++){
        grad[i] = (int)sqrt((grad_x[i] * grad_x[i]) + (grad_y[i] * grad_y[i]));
    }
}

void approximateAngle(short int*& grad_x, short int*& grad_y, int rows, int columns, short int*& angle){
    angle = new short int [rows * columns];
    float temp;
    for(int i = 0; i < (rows*columns); i++){
        temp = atan2((double)grad_y[i],(double)grad_x[i]);
        temp *= (180/M_PI);
        if(temp < 0){
            temp = 360 + temp;
        }
        cout << temp << endl;
        if((temp >= 22.5 && temp < 67.5) || (temp >= 202.5 && temp < 247.5)){
            angle[i] = 45;
        }
        else if((temp >= 112.5 && temp < 157.5) || (temp >= 292.5 && temp < 337.5)){
            angle[i] = 135;
        }
        else if((temp >= 67.5 && temp < 112.5) || (temp >= 247.5 && temp < 292.5)){
            angle[i] = 90;
        }
        else{
            angle[i] = 0;
        }
    }
}