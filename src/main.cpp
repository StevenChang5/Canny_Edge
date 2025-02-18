#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

#include <src/utils.h>
#include <src/cuda.h>

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

        canny(frames[i].data, sigma, minVal, maxVal, frames[i].rows, frames[i].cols);
    }

    return 0;
}