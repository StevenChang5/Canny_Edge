#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

#include <src/utils.h>

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
    /***********************************************************
     * Retrieve command line arguments
    ***********************************************************/
    if(argc < 4){
        fprintf(stderr, "USAGE: %s sigma minVal maxVal\n", argv[0]);
        fprintf(stderr, "   sigma: Standard deviation used for the gaussian blurring kernel\n");
        fprintf(stderr, "   minVal: The minimum threshold value used for hysteresis\n");
        fprintf(stderr, "           Must be in the range of [0,255]\n");
        fprintf(stderr, "   maxVal: The maximum threshold value used for hysteresis\n");
        fprintf(stderr, "           Must be in the range of [0,255]\n");
        exit(0);
    }

    if(atof(argv[2]) >= atof(argv[3])){
        fprintf(stderr, "ERROR: minVal must be less than maxVal\n");
        exit(0);
    }

    float sigma = atof(argv[1]);
    int minVal = atof(argv[2]);
    int maxVal = atof(argv[3]);

    VideoCapture cap;

    if(!cap.open(0)){
        cout << "ERROR: Failed to open camera" << endl;
        return -1;
    }
    
    Mat frame, gray_frame;
    vector<Mat> frames; 
    unsigned char *img;         // Raw image
    short int* smoothed_img;    // Image blurred by a Gaussian filter
    short int* magnitude;       // Magnitude of edges, calculated as sqrt(grad_x^2 + grad_y^2)
    short int* angle;           // Angle/direction of edges, calculated as arctan2(grad_y, grad_x)
    short int* nonmaximal;      // Edges w/ nonmaximal suppression applied to neighbors in angle direction


    /***********************************************************
     * Display camera feed, wait for spacebar to be pressed
    ***********************************************************/
    while(true){
        cap >> frame;
        if(frame.empty()){
            break;
        }
        imshow("Camera Feed", frame);
        if(waitKey(10)==32){
            break;
        }
    }
    
    /***********************************************************
     * Capture frames from the camera
    ***********************************************************/

    /***********************************************************
     * TODO: make the number of frames captured adjustable
    ***********************************************************/
    while(frames.size() < 1){
        cap >> frame; 
        cvtColor(frame, gray_frame, COLOR_BGR2GRAY);
        frames.push_back(gray_frame.clone());
    }

    /***********************************************************
     * Perform canny edge detection on capture frames
    ***********************************************************/
    for(int i = 0; i < frames.size(); i++){

        // Display image before any processing
        imshow("Camera Feed", frames[i]);
        waitKey(0);

        auto start = chrono::high_resolution_clock::now();
        img = frames[i].data;

        // Apply gaussian blurring
        gaussian(img,sigma,frames[i].rows,frames[i].cols,smoothed_img);

        if(STEPS){
            Mat smoothedMat(frames[i].rows,frames[i].cols, CV_16S, smoothed_img);
            Mat smoothed_display;

            normalize(smoothedMat, smoothed_display, 0, 255, NORM_MINMAX);
            smoothed_display.convertTo(smoothed_display, CV_8U);

            imshow("Gaussian Smoothed Image", smoothed_display);
            waitKey(0);
        }

        // Use sobel operator to find magintude and direction of gradient
        sobelOperator(smoothed_img, frames[i].rows, frames[i].cols, magnitude, angle);

        if(STEPS){
            Mat gradientMat(frames[i].rows,frames[i].cols, CV_16S, magnitude);
            Mat gradient_display;
            normalize(gradientMat, gradient_display, 0, 255, NORM_MINMAX);
            gradient_display.convertTo(gradient_display, CV_8U);

            imshow("Edge Image", gradient_display);
            waitKey(0);
        }

        // Apply nonmaximal suppression to sharpen edges
        nonmaximalSuppression(magnitude, angle, frames[i].rows, frames[i].cols, nonmaximal);

        if(STEPS){
            Mat suppressMat(frames[i].rows,frames[i].cols, CV_16S, nonmaximal);
            Mat suppress_display;
            normalize(suppressMat, suppress_display, 0, 255, NORM_MINMAX);
            suppress_display.convertTo(suppress_display, CV_8U);

            imshow("Nonmaximal Image", suppress_display);
            waitKey(0);
        }

        // Use hysteresis to keep pixels with intensities within the given thresholds
        hysteresis(nonmaximal, frames[i].rows, frames[i].cols, minVal, maxVal);
        auto stop = chrono::high_resolution_clock::now();
        // Display final image with canny edge detection applied to it
        Mat finalMat(frames[i].rows,frames[i].cols, CV_16S, nonmaximal);
        Mat final_display;
        normalize(finalMat, final_display, 0, 255, NORM_MINMAX);
        final_display.convertTo(final_display, CV_8U);
        imshow("Nonmaximal Image", final_display);
        waitKey(0);

        chrono::duration<double> duration = stop - start;
        cout << "Execution time: " << duration.count() << " seconds\n";

        delete[] magnitude;
        delete[] angle;
        delete[] nonmaximal;
    }

    return 0;
}