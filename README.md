# Canny_Edge
The purpose of this repository is to implement the steps of canny edge detection. 

## Requirements
* [cmake](https://cmake.org/download/)
* [OpenCV](https://opencv.org/get-started/)

## Usage
To build the program, run the following commands:
```
cmake -S . -B build
cmake --build build
```
The program can then be run by the following:
```
cd build/src
./Main {sigma} {minVal} {maxVal}
```
**Parameters:**
* sigma: the standard deviation used by the gaussian blurring function to generate the gaussian kernel
* minVal: the minimum value for a pixel during the hysteresis step
* maxVal: the maximum value for a pixel during the hysteresis step
