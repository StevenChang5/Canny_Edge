#include "utils.h"

#include <iostream>
#include <cmath>
#include <queue>

using namespace std;

/*************************************************************************
 * NOTE: pixels of Mat.data are represented as a single array, and access
 *       to a pixel at (r, c) is given by Mat.data[r * columns + c]
 ************************************************************************/

/*************************************************************************
 * FUNCTION: performs gaussian blurring on the given image
 * PARAMETERS:
 *      img: input in the form of grayscale image, 1D array from Mat.data 
 *      sigma: standard deviation used in gaussian kernel
 *      rows: height of input/output image
 *      columns: width of input/output image
 *      result: output in the form of grayscale image, 1D array
 ************************************************************************/
void gaussian(unsigned char* img, float sigma, int rows, int columns, short int*& result){
    float* kernel;
    int window;

    createGaussianKernel(kernel,sigma,&window);

    int center = window/2;
    float* temp_img = new float[rows*columns];
    result = new short int[rows*columns];

    // Blur in the x direction
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

    // Blur in the y direction
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
            result[j * columns + i] = (short int)(sum/count);
        }   
    }

    delete[] temp_img;
    delete[] kernel;
}

/*************************************************************************
 * FUNCTION: creates gaussian kernel used for gaussian blurring
 * PARAMETERS:
 *      kernel: pointer used to store gaussian kernel
 *      sigma: standard deviation of gaussian distribution
 *      window: size of gaussian kernel as a function of sigma
 ************************************************************************/
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

/*************************************************************************
 * FUNCTION: calculates derivative of the image in the x and y directions
 * PARAMETERS:
 *      img: input image, smoothed by gaussian blurring
 *      rows: height of input/output image
 *      columns: width of input/output image
 *      grad_x: gradient of each pixel in the x direction
 *      grad_y: gradient of each pixel in the y direction
 ************************************************************************/
void calculateXYGradient(short int*& img, int rows, int columns, short int*& grad_x, short int*& grad_y){
    grad_x = new short int[rows * columns];
    grad_y = new short int[rows * columns];

    // Gradient in the x direction, filter in the form of:
    // -1   0   1
    // -2   0   2
    // -1   0   1
    for(int r = 0; r < rows; r++){
        int pos = r * columns;
        // Leftmost column, all rows; pad out of bounds with border values
        grad_x[pos] = (2 * img[pos+1]) - (2 * img[pos]);
        // include row above
        if(r != rows-1){
            grad_x[pos] += (img[pos+columns+1] - img[pos+columns]);
        }
        // include row below
        if(r != 0){
            grad_x[pos] += (img[pos-columns+1] - img[pos-columns]);
        }
        pos++;

        // Middle, non-border pixels
        for(int c = 1; c < columns-1; c++, pos++){
            grad_x[pos] = (2 * img[pos+1]) - (2 * img[pos-1]);
            if(r != rows-1){
                grad_x[pos] += (img[pos+columns+1] - img[pos+columns-1]);
            }
            if(r != 0){
                grad_x[pos] += (img[pos-columns+1] - img[pos-columns-1]);
            }
        }

        // Rightmost column, all rows; pad out of bounds with border values
        grad_x[pos] = (2 * img[pos]) - (2 * img[pos-1]);
        // include row below
        if(r != rows-1){
            grad_x[pos] += (img[pos+columns] - img[pos+columns-1]);
        }
        // include row above
        if(r != 0){
            grad_x[pos] += (img[pos-columns] - img[pos-columns-1]);
        }
    }

    // Gradient in the y direction, filter in the form of:
    //  1   2   1
    //  0   0   0
    // -1  -2  -1
    for(int c = 0; c < columns; c++){
        int pos = c;
        // Topmost row, all columns; pad out of bounds with border values
        grad_y[pos] = (2 * img[pos+columns]) - (2 * img[pos]);
        if(c != columns-1){
            grad_y[pos] += (img[pos+columns+1]-img[pos+1]);
        }
        if(c != 0){
            grad_y[pos] += (img[pos+columns-1]-img[pos-1]);
        }
        pos += columns;

        // Middle, nonborder pixels
        for(int r = 1; r < rows-1; r++, pos+=columns){
            grad_y[pos] = (2*img[pos+columns]) - (2*img[pos-columns]);
            if(c != columns-1){
                grad_y[pos] += (img[pos+columns+1]-img[pos-columns+1]);
            }
            if(c != 0){
                grad_y[pos] += (img[pos+columns-1]-img[pos-columns-1]);
            }
        }

        // Bottommost row, all columns; pad out of bounds with border values
        grad_y[pos] = (2*img[pos]) - (2*img[pos-columns]);
        if(c != columns-1){
            grad_y[pos] += (img[pos+1] - img[pos-columns+1]);
        }
        if(c != 0){
            grad_y[pos] += (img[pos-1]-img[pos-columns-1]);
        }
    }
}

/*************************************************************************
 * FUNCTION: calculates the gradient direction and magnitude using
 *      gradient in the x and y direction
 * PARAMETERS:
 *      img: input image, smoothed by gaussian blurring
 *      grad_x: gradient in the x direction
 *      grad_y: gradient in the y direction
 *      rows: height of input/output image
 *      columns: width of input/output image
 *      magnitude: magnitude of combined x and y gradient
 *      angle: direction of combined x and y gradient
 ************************************************************************/
void sobelOperator(short int*& img, int rows, int columns, short int*& magnitude, short int*& angle){
    short int* grad_x;          // Gradient in the x direction
    short int* grad_y;          // Gradient in the y direction
    magnitude = new short int[rows * columns];
    angle = new short int[rows * columns];

    calculateXYGradient(img, rows, columns, grad_x, grad_y);

    float temp_angle;
    for(int i = 0; i < (rows * columns); i++){
        // Calculate magnitude of gradient at every pixel
        magnitude[i] = (int)sqrt((grad_x[i] * grad_x[i]) + (grad_y[i] * grad_y[i]));

        // Calculate angle of gradient at every pixel
        temp_angle = atan2((double)grad_y[i],(double)grad_x[i]);
        temp_angle *= (180/PI);
        if(temp_angle < 0){
            temp_angle = 360 + temp_angle;
        }
        if((temp_angle >= 22.5 && temp_angle < 67.5) || (temp_angle >= 202.5 && temp_angle < 247.5)){
            angle[i] = 45;
        }
        else if((temp_angle >= 112.5 && temp_angle < 157.5) || (temp_angle >= 292.5 && temp_angle < 337.5)){
            angle[i] = 135;
        }
        else if((temp_angle >= 67.5 && temp_angle < 112.5) || (temp_angle >= 247.5 && temp_angle < 292.5)){
            angle[i] = 90;
        }
        else{
            angle[i] = 0;
        }
    }
    delete[] grad_x;
    delete[] grad_y;
}

/*************************************************************************
 * FUNCTION: sets pixels to 0 if the neighbors in the direction of their
 *      gradient is greater than them
 * PARAMETERS:
 *      magnitude: magnitude of calculated gradient
 *      angle: direction of calculated magnitude
 *      rows: height of input/output image
 *      columns: width of input/output image
 *      result: gradient with only maximal pixels
 ************************************************************************/
void nonmaximalSuppression(short int*& magnitude, short int *& angle, int rows, int columns, short int*& result){
    bool max;
    result = new short int[rows*columns];
    for(int i = 0; i < rows * columns; i++){
        max = true;
        if(angle[i] == 0){
            int left = i - 1;
            int right = i + 1;

            if((i%columns) > 0){
                if(magnitude[i] <= magnitude[left]){max = false;}
            }
            if(i%columns < columns-1){
                if(magnitude[i] <= magnitude[right]){max = false;}
            }
            if(max){result[i] = magnitude[i];}
            else{result[i] = NOEDGE;}
        }   
        else if(angle[i] == 45){
            int upRight = i + 1 - columns;
            int downLeft = i - 1 + columns;

            if((i%columns < columns-1) && (i >= rows)){
                if(magnitude[i] <= magnitude[upRight]){max = false;}
            }
            if((i%columns > 0) && (i < (rows*columns)-rows)){
                if(magnitude[i] <= magnitude[downLeft]){max = false;}
            }
            if(max){result[i] = magnitude[i];}
            else{result[i] = NOEDGE;}
        }
        else if(angle[i] == 90){
            int up = i - columns;
            int down = i + columns;

            if(i >= rows){
                if(magnitude[i] <= magnitude[up]){max = false;}
            }
            if(i < (rows*columns)-rows){
                if(magnitude[i] <= magnitude[down]){max = false;}
            }
            if(max){result[i] = magnitude[i];}
            else{result[i] = NOEDGE;}
        } 
        else if(angle[i] == 135){
            int upLeft = i - 1 - columns;
            int downRight = i + 1 + columns;

            if((i%columns > 0) && (i >= rows)){
                if(magnitude[i] <= magnitude[upLeft]){max = false;}
            }
            if((i%columns < columns-1) && (i < (rows*columns)-rows)){
                if(magnitude[i] <= magnitude[downRight]){max = false;}
            }
            if(max){result[i] = magnitude[i];}
            else{result[i] = NOEDGE;}
        }
    }
}

/*************************************************************************
 * FUNCTION: sets pixels to 0 if the neighbors in the direction of their
 *      gradient is greater than them
 * PARAMETERS:
 *      edgeCandidates: gradient with only maximal pixels
 *      rows: height of input/output image
 *      columns: width of input/output image
 *      minVal: minimum threshold value for pixels, any pixel with a
 *          gradient magnitude less than this will be set to NOEDGE
 *      maxVal: maximum threshold value for pixels, any pixel with a
 *          gradient magnitude greater than this will be set to EDGE
 ************************************************************************/
void hysteresis(short int*& edgeCandidates, int rows, int columns, int minVal, int maxVal){
    bool* visited = new bool [rows * columns];
    fill_n(visited, rows*columns,false);
    
    // Filter out any pixels below minimum threshold, find all pixels connected to strong edges
    for(int i = 0; i < rows*columns; i++){
        if(edgeCandidates[i] < minVal){
            edgeCandidates[i] = NOEDGE;
        }
        else if(edgeCandidates[i] >= maxVal){
            findEdgePixels(edgeCandidates, visited, i, minVal, maxVal, rows, columns);
        }
    }
    // Filter out any pixels between minimum and maximum threshold that are not connected to strong edges
    for(int i = 0; i < rows*columns; i++){
        if(edgeCandidates[i] < maxVal){
            edgeCandidates[i] = NOEDGE;
        }
    }
    delete[] visited;
}

/*************************************************************************
 * FUNCTION: finds all pixels with magnitude greater than minVal that are
 *      connected to pixel at [start]
 * PARAMETERS:
 *      edgeCandidates: gradient with only maximal pixels
 *      visited: array of all pixels with true representing the pixels
 *          has been visited by the algorithm
 *      start: strong edge to start the search from
 *      columns: width of input/output image
 *      minVal: minimum threshold value for pixels, any pixel with a
 *          gradient magnitude less than this will be set to NOEDGE
 *      maxVal: maximum threshold value for pixels, any pixel with a
 *          gradient magnitude greater than this will be set to EDGE
 *      rows: height of input/output image
 *      columns: width of input/output image
 ************************************************************************/
void findEdgePixels(short int*& edgeCandidates, bool*& visited, int start, int minVal, int maxVal, int rows, int columns){
    if(visited[start] == true){return;}
    queue<int> pixels;
    int current; 

    pixels.push(start);
    while(!pixels.empty()){
        current = pixels.front();
        edgeCandidates[current] = EDGE;
        if(current%columns > 0){
            // Check pixel to the bottom left
            if(current < (rows*columns)-rows){
                if(edgeCandidates[current+columns - 1] >= minVal && !visited[current+columns-1]){
                    pixels.push(current+columns-1);
                    visited[current+columns-1] = true;
                }
            }
            // Check pixel to the top left
            if(current >= rows){
                if(edgeCandidates[current-columns-1] >= minVal && !visited[current-columns-1]){
                    pixels.push(current-columns-1);
                    visited[current-columns-1] = true;
                }
            }
            // Check pixel to the left
            if(edgeCandidates[current-1] >= minVal && !visited[current-1]){
                pixels.push(current-1);
                visited[current-1] = true;
            }
        }
        if(current%columns < columns-1){
            // Check pixel to the bottom right
            if(current < (rows*columns)-rows){
                if(edgeCandidates[current+columns+1] >= minVal && !visited[current+columns+1]){
                    pixels.push(current+columns+1);
                    visited[current+columns+1] = true;
                }
            }
            // Check pixel to the top right
            if(current >= rows){
                if(edgeCandidates[current-columns+1] >= minVal && !visited[current-columns+1]){
                    pixels.push(current-columns+1);
                    visited[current-columns+1] = true;
                }
            }
            // Check pixel to the right
            if(edgeCandidates[current+1] >= minVal && !visited[current+1]){
                pixels.push(current+1);
                visited[current+1] = true;
            }
        }
        // Check pixel below
        if(current < (rows*columns)-rows){
            if(edgeCandidates[current+columns] >= minVal && !visited[current+columns]){
                pixels.push(current+columns);
                visited[current+columns] = true;
            }
        }
        // Check pixel above
        if(current >= rows){
            if(edgeCandidates[current-columns] >= minVal && !visited[current-columns]){
                pixels.push(current-columns);
                visited[current-columns] = true;
            }
        }
        pixels.pop();
    }
}