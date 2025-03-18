#include "COOMatrixGPU.cuh"

int main(){
    size_t nRows = 10, nCols = 10;
    float* mat = new float[nRows * nCols];
    COOMatrixGPU(mat, nRows, nCols);
}