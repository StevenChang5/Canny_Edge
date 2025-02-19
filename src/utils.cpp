#include "utils.h"
#include <opencv2/opencv.hpp>

#include <chrono>
#include <iostream>
#include <cmath>
#include <queue>

using namespace cv;
using namespace std;

/*************************************************************************
 * NOTE: pixels of Mat.data are represented as a single array, and access
 *       to a pixel at (r, c) is given by Mat.data[r * width + c]
 ************************************************************************/

/*************************************************************************
 * FUNCTION: performs gaussian blurring on the given image
 * PARAMETERS:
 *      img: input in the form of grayscale image, 1D array from Mat.data 
 *      sigma: standard deviation used in gaussian kernel
 *      height: height of input/output image
 *      width: width of input/output image
 *      result: output in the form of grayscale image, 1D array
 ************************************************************************/
void gaussian(unsigned char*& img, float sigma, int height, int width, short int*& result){
    float* kernel;
    int window;

    createGaussianKernel(kernel,sigma,&window);

    int center = window/2;
    float* temp_img = new float[height*width];
    result = new short int[height*width];

    // Blur in the x direction
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            float sum = 0;
            float count = 0;
            for(int k = -center; k < (center+1); k++){
                if((j+k >= 0) and (j+k < width)){
                    sum += (float(img[i * width + (j+k)]) * kernel[center+k]);
                    count += kernel[center+k];
                }
            }
            temp_img[i * width + j] = (sum/count);
        }
    }

    // Blur in the y direction
    for(int i = 0; i < width; i++){
        for(int j = 0; j < height; j++){
            float sum = 0;
            float count = 0; 
            for(int k = -center; k<(center+1); k++){
                if((j+k >= 0) and (j+k < height)){
                    sum += (float(temp_img[(j+k) * width + i]) * kernel[center+k]);
                    count += kernel[center+k];
                }
            }
            result[j * width + i] = (short int)(sum/count);
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
 *      height: height of input/output image
 *      width: width of input/output image
 *      grad_x: gradient of each pixel in the x direction
 *      grad_y: gradient of each pixel in the y direction
 ************************************************************************/
void calculateXYGradient(short int*& img, int height, int width, short int*& grad_x, short int*& grad_y){
    grad_x = new short int[height * width];
    grad_y = new short int[height * width];

    // Gradient in the x direction, filter in the form of:
    // -1   0   1
    // -2   0   2
    // -1   0   1
    for(int r = 0; r < height; r++){
        int pos = r * width;
        // Leftmost column, all height; pad out of bounds with border values
        grad_x[pos] = (2 * img[pos+1]) - (2 * img[pos]);
        // include row above
        if(r != height-1){
            grad_x[pos] += (img[pos+width+1] - img[pos+width]);
        }
        // include row below
        if(r != 0){
            grad_x[pos] += (img[pos-width+1] - img[pos-width]);
        }
        pos++;

        // Middle, non-border pixels
        for(int c = 1; c < width-1; c++, pos++){
            grad_x[pos] = (2 * img[pos+1]) - (2 * img[pos-1]);
            if(r != height-1){
                grad_x[pos] += (img[pos+width+1] - img[pos+width-1]);
            }
            if(r != 0){
                grad_x[pos] += (img[pos-width+1] - img[pos-width-1]);
            }
        }

        // Rightmost column, all height; pad out of bounds with border values
        grad_x[pos] = (2 * img[pos]) - (2 * img[pos-1]);
        // include row below
        if(r != height-1){
            grad_x[pos] += (img[pos+width] - img[pos+width-1]);
        }
        // include row above
        if(r != 0){
            grad_x[pos] += (img[pos-width] - img[pos-width-1]);
        }
    }

    // Gradient in the y direction, filter in the form of:
    //  1   2   1
    //  0   0   0
    // -1  -2  -1
    for(int c = 0; c < width; c++){
        int pos = c;
        // Topmost row, all width; pad out of bounds with border values
        grad_y[pos] = (2 * img[pos+width]) - (2 * img[pos]);
        if(c != width-1){
            grad_y[pos] += (img[pos+width+1]-img[pos+1]);
        }
        if(c != 0){
            grad_y[pos] += (img[pos+width-1]-img[pos-1]);
        }
        pos += width;

        // Middle, nonborder pixels
        for(int r = 1; r < height-1; r++, pos+=width){
            grad_y[pos] = (2*img[pos+width]) - (2*img[pos-width]);
            if(c != width-1){
                grad_y[pos] += (img[pos+width+1]-img[pos-width+1]);
            }
            if(c != 0){
                grad_y[pos] += (img[pos+width-1]-img[pos-width-1]);
            }
        }

        // Bottommost row, all width; pad out of bounds with border values
        grad_y[pos] = (2*img[pos]) - (2*img[pos-width]);
        if(c != width-1){
            grad_y[pos] += (img[pos+1] - img[pos-width+1]);
        }
        if(c != 0){
            grad_y[pos] += (img[pos-1]-img[pos-width-1]);
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
 *      height: height of input/output image
 *      width: width of input/output image
 *      magnitude: magnitude of combined x and y gradient
 *      angle: direction of combined x and y gradient
 ************************************************************************/
void sobelOperator(short int*& img, int height, int width, short int*& magnitude, short int*& angle){
    short int* grad_x;          // Gradient in the x direction
    short int* grad_y;          // Gradient in the y direction
    magnitude = new short int[height * width];
    angle = new short int[height * width];

    calculateXYGradient(img, height, width, grad_x, grad_y);

    float temp_angle;
    for(int i = 0; i < (height * width); i++){
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
    delete[] img;
}

/*************************************************************************
 * FUNCTION: sets pixels to 0 if the neighbors in the direction of their
 *      gradient is greater than them
 * PARAMETERS:
 *      magnitude: magnitude of calculated gradient
 *      angle: direction of calculated magnitude
 *      height: height of input/output image
 *      width: width of input/output image
 *      result: gradient with only maximal pixels
 ************************************************************************/
void nonmaximalSuppression(short int*& magnitude, short int *& angle, int height, int width, short int*& result){
    bool max;
    result = new short int[height*width];
    for(int i = 0; i < height * width; i++){
        max = true;
        if(angle[i] == 0){
            int left = i - 1;
            int right = i + 1;

            if((i%width) > 0){
                if(magnitude[i] <= magnitude[left]){max = false;}
            }
            if(i%width < width-1){
                if(magnitude[i] <= magnitude[right]){max = false;}
            }
            if(max){result[i] = magnitude[i];}
            else{result[i] = NOEDGE;}
        }   
        else if(angle[i] == 45){
            int upRight = i + 1 - width;
            int downLeft = i - 1 + width;

            if((i%width < width-1) && (i - width >= 0)){
                if(magnitude[i] <= magnitude[upRight]){max = false;}
            }
            if((i%width > 0) && (i + width < (height*width))){
                if(magnitude[i] <= magnitude[downLeft]){max = false;}
            }
            if(max){result[i] = magnitude[i];}
            else{result[i] = NOEDGE;}
        }
        else if(angle[i] == 90){
            int up = i - width;
            int down = i + width;

            if(i - width >= 0){
                if(magnitude[i] <= magnitude[up]){max = false;}
            }
            if(i + width < (height*width)){
                if(magnitude[i] <= magnitude[down]){max = false;}
            }
            if(max){result[i] = magnitude[i];}
            else{result[i] = NOEDGE;}
        } 
        else if(angle[i] == 135){
            int upLeft = i - 1 - width;
            int downRight = i + 1 + width;

            if((i%width > 0) && (i - width >= 0)){
                if(magnitude[i] <= magnitude[upLeft]){max = false;}
            }
            if((i%width < width-1) && (i + width < (height*width))){
                if(magnitude[i] <= magnitude[downRight]){max = false;}
            }
            if(max){result[i] = magnitude[i];}
            else{result[i] = NOEDGE;}
        }
    }
    delete[] magnitude;
    delete[] angle;
}

/*************************************************************************
 * FUNCTION: sets pixels to 0 if the neighbors in the direction of their
 *      gradient is greater than them
 * PARAMETERS:
 *      edgeCandidates: gradient with only maximal pixels
 *      height: height of input/output image
 *      width: width of input/output image
 *      minVal: minimum threshold value for pixels, any pixel with a
 *          gradient magnitude less than this will be set to NOEDGE
 *      maxVal: maximum threshold value for pixels, any pixel with a
 *          gradient magnitude greater than this will be set to EDGE
 ************************************************************************/
void hysteresis(short int*& edgeCandidates, int height, int width, int minVal, int maxVal){
    bool* visited = new bool [height * width];
    fill_n(visited, height*width,false);
    
    // Filter out any pixels below minimum threshold, find all pixels connected to strong edges
    for(int i = 0; i < height*width; i++){
        if(edgeCandidates[i] < minVal){
            edgeCandidates[i] = NOEDGE;
        }
        else if(edgeCandidates[i] >= maxVal){
            findEdgePixels(edgeCandidates, visited, i, minVal, maxVal, height, width);
        }
    }
    // Filter out any pixels between minimum and maximum threshold that are not connected to strong edges
    for(int i = 0; i < height*width; i++){
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
 *      width: width of input/output image
 *      minVal: minimum threshold value for pixels, any pixel with a
 *          gradient magnitude less than this will be set to NOEDGE
 *      maxVal: maximum threshold value for pixels, any pixel with a
 *          gradient magnitude greater than this will be set to EDGE
 *      height: height of input/output image
 *      width: width of input/output image
 ************************************************************************/
void findEdgePixels(short int*& edgeCandidates, bool*& visited, int start, int minVal, int maxVal, int height, int width){
    if(visited[start] == true){return;}
    queue<int> pixels;
    int current; 

    pixels.push(start);
    while(!pixels.empty()){
        current = pixels.front();
        edgeCandidates[current] = EDGE;
        if(current%width > 0){
            // Check pixel to the bottom left
            if(current + width < (height*width)){
                if(edgeCandidates[current+width - 1] >= minVal && !visited[current+width-1]){
                    pixels.push(current+width-1);
                    visited[current+width-1] = true;
                }
            }
            // Check pixel to the top left
            if(current - width > 0){
                if(edgeCandidates[current-width-1] >= minVal && !visited[current-width-1]){
                    pixels.push(current-width-1);
                    visited[current-width-1] = true;
                }
            }
            // Check pixel to the left
            if(edgeCandidates[current-1] >= minVal && !visited[current-1]){
                pixels.push(current-1);
                visited[current-1] = true;
            }
        }
        if(current%width < width-1){
            // Check pixel to the bottom right
            if(current + width < (height*width)){
                if(edgeCandidates[current+width+1] >= minVal && !visited[current+width+1]){
                    pixels.push(current+width+1);
                    visited[current+width+1] = true;
                }
            }
            // Check pixel to the top right
            if(current - width > 0){
                if(edgeCandidates[current-width+1] >= minVal && !visited[current-width+1]){
                    pixels.push(current-width+1);
                    visited[current-width+1] = true;
                }
            }
            // Check pixel to the right
            if(edgeCandidates[current+1] >= minVal && !visited[current+1]){
                pixels.push(current+1);
                visited[current+1] = true;
            }
        }
        // Check pixel below
        if(current + width < (height*width)){
            if(edgeCandidates[current+width] >= minVal && !visited[current+width]){
                pixels.push(current+width);
                visited[current+width] = true;
            }
        }
        // Check pixel above
        if(current - width >= 0){
            if(edgeCandidates[current-width] >= minVal && !visited[current-width]){
                pixels.push(current-width);
                visited[current-width] = true;
            }
        }
        pixels.pop();
    }
}

void canny(unsigned char* img, float sigma, int minVal, int maxVal, int height, int width, bool steps){
    short int* smoothed_img;    // Image blurred by a Gaussian filter
    short int* magnitude;       // Magnitude of edges, calculated as sqrt(grad_x^2 + grad_y^2)
    short int* angle;           // Angle/direction of edges, calculated as arctan2(grad_y, grad_x)
    short int* nonmaximal;      // Edges w/ nonmaximal suppression applied to neighbors in angle direction

    auto start = chrono::high_resolution_clock::now();

    // Apply gaussian blurring
    gaussian(img,sigma,height,width,smoothed_img);

    if(steps){
        Mat smoothedMat(height,width, CV_16S, smoothed_img);
        Mat smoothed_display;

        normalize(smoothedMat, smoothed_display, 0, 255, NORM_MINMAX);
        smoothed_display.convertTo(smoothed_display, CV_8U);

        imshow("Gaussian Smoothed Image", smoothed_display);
        waitKey(0);
    }

    // Use sobel operator to find magintude and direction of gradient
    sobelOperator(smoothed_img, height, width, magnitude, angle);

    if(steps){
        Mat gradientMat(height,width, CV_16S, magnitude);
        Mat gradient_display;
        normalize(gradientMat, gradient_display, 0, 255, NORM_MINMAX);
        gradient_display.convertTo(gradient_display, CV_8U);

        imshow("Edge Image", gradient_display);
        waitKey(0);
    }

    // Apply nonmaximal suppression to sharpen edges
    nonmaximalSuppression(magnitude, angle, height, width, nonmaximal);

    if(steps){
        Mat suppressMat(height,width, CV_16S, nonmaximal);
        Mat suppress_display;
        normalize(suppressMat, suppress_display, 0, 255, NORM_MINMAX);
        suppress_display.convertTo(suppress_display, CV_8U);

        imshow("Nonmaximal Image", suppress_display);
        waitKey(0);
    }

    // Use hysteresis to keep pixels with intensities within the given thresholds
    hysteresis(nonmaximal, height, width, minVal, maxVal);
    auto stop = chrono::high_resolution_clock::now();
    // Display final image with canny edge detection applied to it
    Mat finalMat(height,width, CV_16S, nonmaximal);
    Mat final_display;
    normalize(finalMat, final_display, 0, 255, NORM_MINMAX);
    final_display.convertTo(final_display, CV_8U);
    imshow("Nonmaximal Image", final_display);
    waitKey(0);

    chrono::duration<double> duration = stop - start;
    cout << "Execution time: " << duration.count() << " seconds\n";

    delete[] nonmaximal;
}