#include "COOMatrixGPU.cuh"
#include "utils.h"
#include "spMV.cuh"
#include <cstring>

int main(){
    size_t nRows = 10, nCols = 10;
    float *mat = new float[nRows * nCols];
    float *x = new float[nCols];
    float *y_cpu = new float[nRows];
    float *y_gpu = new float[nRows];

    initVector(mat, nRows * nCols);
    initVector(x, nCols);
    memset(y_cpu, 0, nRows * sizeof(float));
    memset(y_gpu, 0, nRows * sizeof(float));
    

    matmul_cpu(mat, nRows, nCols, x, nCols, 1, y_cpu);
    COOMatrixGPU matCOO(mat, nRows, nCols);
    spmvCoo_gpu(matCOO, nRows, nCols, x, y_gpu);
    
    compareArrays(y_cpu, y_gpu, nRows);

    delete[] x;
    delete[] y_cpu;
    delete[] y_gpu;
    delete[] mat;
}