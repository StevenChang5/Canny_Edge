#include <opencv2/opencv.hpp>
#include <iostream>

#include <src/utils.h>

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
    float sigma = atof(argv[1]);

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
        for(int i = 0; i<(frames[i].rows * frames[i].cols);i++){
            cout << smoothed_img[i] << " ";
        }

        Mat smoothed_display;
        normalize(smoothedMat, smoothed_display, 0, 255, NORM_MINMAX);
        smoothed_display.convertTo(smoothed_display, CV_8U);

        imshow("Gaussian Smoothed Image", smoothed_display);
        waitKey(0);

        // Todo: Derivative calculation
    }
    return 0;
}