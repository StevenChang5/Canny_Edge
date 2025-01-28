#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main() {
    VideoCapture cap;

    if(!cap.open(0)){
        cout << "Failed to open camera" << endl;
    }
    
    Mat frame, gray_frame;
    vector<Mat> frames; 
    unsigned char *img;

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
        // TODO: Gaussian blur image
    }
    return 0;
}