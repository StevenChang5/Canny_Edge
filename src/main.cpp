#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <vector>

#include <src/utils.h>
#include <src/cuda.h>

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
    /***********************************************************
     * Retrieve command line arguments
    ***********************************************************/
    float sigma;
    int minVal;
    int maxVal;
    bool use_cuda = false;
    bool show_steps = false;
    vector<string> values;

    for(int i = 1; i < argc; i++){
        string arg = argv[i];

        if(arg == "-c"){
            use_cuda = true;
        }
        else if(arg == "-s"){
            show_steps = true;
        }
        else{
            values.push_back(arg);
        }
    }

    if(values.size() != 3){
        fprintf(stderr, "USAGE: %s sigma minVal maxVal\n", argv[0]);
        fprintf(stderr, "   sigma: Standard deviation used for the gaussian blurring kernel\n");
        fprintf(stderr, "   minVal: The minimum threshold value used for hysteresis\n");
        fprintf(stderr, "           Must be in the range of [0,255]\n");
        fprintf(stderr, "   maxVal: The maximum threshold value used for hysteresis\n");
        fprintf(stderr, "           Must be in the range of [0,255]\n");
        exit(0);
    }

    sigma = stof(values[0]);
    minVal = stoi(values[1]);
    maxVal = stoi(values[2]);
    

    if(maxVal <= minVal){
        fprintf(stderr, "ERROR: minVal must be less than maxVal\n");
        exit(0);
    }

    if(minVal < 0 or minVal > 255){
        fprintf(stderr, "ERROR: minVal must be in the range of [0,255]");
        exit(0);
    }

    if(maxVal < 0 or maxVal > 255){
        fprintf(stderr, "ERROR: maxVal must be in the range of [0,255]");
        exit(0);
    }

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

        if(use_cuda){
            cuda_canny(frames[i].data, sigma, minVal, maxVal, frames[i].rows, frames[i].cols, show_steps);
        }
        else{
            canny(frames[i].data, sigma, minVal, maxVal, frames[i].rows, frames[i].cols, show_steps);
        }
        
    }

    return 0;
}