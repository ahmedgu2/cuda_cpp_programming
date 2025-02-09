#include <iostream>
#include <random>
#include <cuda_runtime.h>
#include <cmath>

// GPU version
__device__ 
float normal_pdf(const float x, const float mean, const float std){
    const float pi = 3.141592653589;
    const float coeff = 1 /  (std * sqrtf(2 * pi));
    const float exponent = - 0.5 * powf((x - mean) / std, 2.0);
    return coeff * expf(exponent);
}

__global__
void gaussianBlur_gpu(
    float **mat,
    const int nRows,
    const int nCols,
    const int kernelSize
){

}
 
// CPU version
float normal_pdf(const float x, const float mean, const float std){
    const float pi = 3.141592653589;
    const float coeff = 1 /  (std * std::sqrt(2 * pi));
    const float exponent = - 0.5 * std::pow((x - mean) / std, 2.0);
    return coeff * std::exp(exponent);
}


void gaussianBlur_cpu(
    float **mat,
    const int nRows,
    const int nCols,
    const int kernelSize
){
    float **kernel;
    std::normal_distribution<float> gaussian(0, 1.f);
    
}

int main(){

    return 0;
}