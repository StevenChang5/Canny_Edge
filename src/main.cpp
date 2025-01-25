#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main() {
    VideoCapture cap;

    if(!cap.open(0)){
        cout << "Failed to open camera" << endl;
    }
    
    Mat frame, grey_frame;
    vector<Mat> frames; 

    while(true){
        cap >> frame;
        if(frame.empty()){
            break;
        }
        imshow(frame);
        // On pressing space, continue
        if(waitkey(10)==32){
            break;
        }
    }
    
    // Convert frames to grayscale, add them to vector of frames for processing
    while frames.size() < 1:
        cap >> frame; 
        cvtColor(frame, grey_frame, COLOR_BGR2GRAY);
        frames.push_back(frame);
    
    for(int i = 0; i < frames.size(); i++){
        unsigned char* image = frames.data();
        // TODO: Gaussian blur image
    }
    return 0;
}