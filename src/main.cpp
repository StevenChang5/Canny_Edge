#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::VideoCapture capture(0);
    
    cv::Mat img;

    capture >> img;
    cv::imshow("Image", img); // Display the image in a window
    cv::waitKey(0);             // Wait for a key press
    return 0;
}