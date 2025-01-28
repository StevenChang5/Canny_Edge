#include "utils.h"

#include <iostream>
#include <cmath>

using namespace std;

void gaussian(unsigned char* img, int k){

    return;
    
}

// float*& has same pointer being used outside/inside function, rather than function
// copying address. Necessary for 'kernel = new float[*window]'
void createGaussianKernel(float*& kernel , float sigma, int* window){
    *window = 1 + 2 * ceil(3 * sigma);

    kernel = new float[*window];
    
    int center = (*window)/2;
    float x;

    for(int i = 0; i < *window; i++){
        x = float(i - center);
        float product = exp(-((x*x)/(2*sigma*sigma)))/(sqrt(6.2831853)*sigma);
        kernel[i] = product;
    }
}