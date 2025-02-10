#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

// GPU version
__device__ 
float normal_pdf2D_gpu(const float x, const float y, const float mean, const float std){
    const float pi = 3.141592653589;
    const float coeff = 1 / (2 * pi * std * std);
    const float exponent = - 0.5 * ((x * x + y * y) / (std * std));
    return coeff * expf(exponent);
}

__global__
void gaussianBlur_gpu(
    float *mat,
    const int nRows,
    const int nCols,
    const int kernelSize
){

}
