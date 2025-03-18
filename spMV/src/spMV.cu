#include "cuda_macros.cuh"
#include "COOMatrix.h"

__global__
void spmv_coo(COOMatrix<float>& mat, float *x, float *y){
    int indx = threadIdx.x + blockDim.x * blockIdx.x;
    if(indx < mat.getNonZeroLength()){
        int row = 
    }
}