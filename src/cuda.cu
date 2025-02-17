// cuda.cu
void createGaussianKernel(float*& kernel , float sigma, int window){
    kernel = new float[window];
    
    int center = (window)/2;
    float x;
    float sum = 0.0;

    for(int i = 0; i < window; i++){
        x = float(i - center);
        float product = exp(-((x*x)/(2*sigma*sigma)))/(sqrt(6.2831853)*sigma);
        kernel[i] = product;
        sum += product;
    }

    for(int i = 0; i < window; i++){
        kernel[i] /= sum;
    }
}

__global__ void gaussian_util(unsigned char* img, float sigma, int window, int rows, int columns, unsigned char* temp_img, short int* result){
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = idx_y * columns + idx_x;
    int stride = blockDim.x * gridDim.x;

    int center = window/2;

    float* kernel; 


    // TODO: Bring to CPU level?
    cudaMallocManaged(&kernel, window*sizeof(float));
    createGaussianKernel(kernel, sigma, window);


    // Blur in the x direction
    for(int i = idx; i < rows * columns; i+= stride){
        float sum = 0;
        float count = 0;
        for(int k = -center; k < (center + 1); k++){
            if(((i+k)%rows >= 0) and (i+k)%rows < rows){
                sum += (float(img[idx + k]) * kernel[center + k]);
                count += kernel[center+k];
            }
            
        }
        temp_img[idx] = sum/count;
    }

    __syncthreads();

    // Blur in the y direction
    for(int i = idx; i < rows * columns; i += stride){
        float sum = 0;
        float count = 0;
        for(int k = -center; k < (center + 1); k++){
            if(((i+(k*columns)) >= columns) and ((i+(k*columns)) < (rows*columns)-columns)){
                sum += (float(temp_img[i+(k*columns)]) * kernel[center+k]);
                count += kernel[center+k];
            }
        }
    }

    __syncthreads();
}

void gaussian(unsigned char* img, float sigma, int rows, int columns, short int*& result){
    unsigned char* temp_img; 
    int window = 1 + 2 * ceil(3 * sigma);

    cudaMallocManaged(&temp_img, rows*columns*sizeof(unsigned char));
    cudaMallocManaged(&result, rows*columns*sizeof(short int));

    gaussian_util<<<1,1>>>(img, sigma, window, rows, columns, temp_img, result);

    cudaFree(temp_img);
}



