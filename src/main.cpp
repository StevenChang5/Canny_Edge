#include <opencv2/opencv.hpp>
#include <iostream>

#include <src/utils.h>

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
    float sigma = atof(argv[1]);
    int minVal = atof(argv[2]);
    int maxVal = atof(argv[3]);

    VideoCapture cap;

    if(!cap.open(0)){
        cout << "Failed to open camera" << endl;
        return -1;
    }
    
    Mat frame, gray_frame;
    vector<Mat> frames; 
    unsigned char *img;
    short int* smoothed_img;

    while(true){
        cap >> frame;
        if(frame.empty()){
            break;
        }
        imshow("Camera Feed", frame);
        // On pressing space, continue
        if(waitKey(10)==32){
            break;
        }
    }
    
    // Convert frames to grayscale, add them to vector of frames for processing
    while(frames.size() < 1){
        cap >> frame; 
        cvtColor(frame, gray_frame, COLOR_BGR2GRAY);
        frames.push_back(gray_frame.clone());
    }

    for(int i = 0; i < frames.size(); i++){
        img = frames[i].data;
        /* 
        GAUSSIAN SMOOTHING
        */
        imshow("Test", frames[i]);
        waitKey(0);
        gaussian(img,sigma,frames[i].rows,frames[i].cols,smoothed_img);

        // For displaying smoothed image
        Mat smoothedMat(frames[i].rows,frames[i].cols, CV_16S, smoothed_img);
        Mat smoothed_display;

        normalize(smoothedMat, smoothed_display, 0, 255, NORM_MINMAX);
        smoothed_display.convertTo(smoothed_display, CV_8U);

        imshow("Gaussian Smoothed Image", smoothed_display);
        waitKey(0);

        // Derivative calculation
        short int* grad_x;
        short int* grad_y;
        short int* grad;

        calculateXYGradient(smoothed_img,frames[i].rows,frames[i].cols,grad_x,grad_y);

        approximateGradient(grad_x,grad_y,frames[i].rows,frames[i].cols,grad);

        // For displaying smoothed image
        Mat gradientMat(frames[i].rows,frames[i].cols, CV_16S, grad);
        Mat gradient_display;
        normalize(gradientMat, gradient_display, 0, 255, NORM_MINMAX);
        gradient_display.convertTo(gradient_display, CV_8U);

        imshow("Edge Image", gradient_display);
        waitKey(0);

        // Non-maximal Suppression
        short int* angle;
        short int* suppress;

        approximateAngle(grad_x, grad_y, frames[i].rows, frames[i].cols, angle);
        
        nonmaximalSuppression(grad, angle, frames[i].rows, frames[i].cols, suppress);
        
        Mat suppressMat(frames[i].rows,frames[i].cols, CV_16S, suppress);
        Mat suppress_display;
        normalize(suppressMat, suppress_display, 0, 255, NORM_MINMAX);
        suppress_display.convertTo(suppress_display, CV_8U);

        imshow("Nonmaximal Image", suppress_display);
        waitKey(0);

        // Hysteresis
        hysteresis(suppress, frames[i].rows, frames[i].cols, minVal, maxVal);

        Mat finalMat(frames[i].rows,frames[i].cols, CV_16S, suppress);
        Mat final_display;
        normalize(finalMat, final_display, 0, 255, NORM_MINMAX);
        final_display.convertTo(final_display, CV_8U);

        imshow("Nonmaximal Image", final_display);
        waitKey(0);

        delete[] grad_x;
        delete[] grad_y;
        delete[] grad;
        delete[] angle;
        delete[] suppress;
    }
    return 0;
}