# Canny_Edge
The purpose of this repository is to implement the steps of canny edge detection. 

## Dependencies
* [cmake](https://cmake.org/download/)
* [OpenCV](https://opencv.org/get-started/)
* [cuda (optional)](https://developer.nvidia.com/cuda-toolkit)

## Usage
To build the program, run the following commands:
```
cmake -S . -B build
cmake --build build
```
To enable cuda during:
```
cmake -D ENABLE_CUDA=ON build
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
* -s: show steps of process
* -c: use cuda functions if cuda is enabled
## Example
<p align="center">
    <img src="figures/example.jpg" alt="Original Image" width="180" height="180"/>
    <img src="figures/gaussian.jpeg" alt="Gaussian Blur" width="180" height="180"/>
    <img src="figures/gradient.jpeg" alt="Gradient Calculation" width="180" height="180"/>
    <img src="figures/nonmaximal.jpeg" alt="Nonmaximal Suppression" width="180" height="180"/>
    <img src="figures/hysteresis.jpeg" alt="Hysteresis Thresholding" width="180" height="180"/>
</p>

1. **Gaussian Blurring:** The image is first changed to a grayscale image. To remove noise that can affect edge detection, a gaussian filter is applied to the image. A gaussian filter is generated as a function of the provided sigma value, and is used to smooth the image.

2. **Gradient Calculation:** The edges were then found by using the Sobel edge detection operator. The derivative in the horizontal and vertical direction were found separately. The resulting derived images were combined to find the edge gradient magnitude and direction using the equations: $G = \sqrt(G_x^2 + G_y^2)$, $\theta = \text{arctan}2(G_y,G_x)$

3. **Nonmaximal Suppression:** Nonmaximal suppression was used to thin the edges. Based on the gradient direction, it was classified as going in the North/South, East/West, Northwest/Southeast, or Northeast/Southwest direction. The gradient of the neighboring pixels in the classified direction were then checked. If the gradient at the pixel was not larger than both of its neighbors, the value of the pixel was set to 0. 

4. **Hysteresis Thresholding:** Lastly, hysteresis thresholding was used to keep strong edges and remove weak edges. Using the provided minVal and maxVal, any edges with a value lower than minVal was set to 0, and any edge higher than maxVal was set to the maximum value (255). Edges that fell between the two threshold values were classified based on if they touched a strong edge. Edges that touch a strong edge were set to the maximum value and were from then on considered as strong edges, while edges that did not touch strong edges were set to 0. 
