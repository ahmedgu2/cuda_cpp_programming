#pragma once

class CSRMatrixGPU{
public:
    CSRMatrixGPU() {}

    __device__ size_t getnRows() const {return nRows;}
    __device__ float getValues(uint32_t indx) {return values[indx];}
    __device__ float getRowPtrs(uint32_t indx) {return rowPtrs[indx];}
    __device__ float getColsIndx(uint32_t indx) {return colsIndx[indx];}
    __host__ __device__ size_t getNumNonZeros() const {
        #ifdef __CUDA_ARCH__:
            return *d_numNonZeros;
        #else
            return h_numNonZeros;
        #endif
    }

private:
    size_t *d_numNonZeros;
    size_t h_numNonZeros;
    size_t nRows, nCols;
    float *values;
    uint32_t *rowPtrs, *colsIndx;
};