#pragma once
#include <cuda_runtime.h>
#include "cuda_macros.cuh"

class CSRMatrixGPU{
public:
    CSRMatrixGPU(float* mat, size_t nRows, size_t nCols){
        this->nRows = nRows;
        this->nCols = nCols;
        // Compute non zeros
        h_numNonZeros = 0;
        for(size_t i = 0; i < nRows; ++i){
            for(size_t j = 0; j < nCols; ++j){
                if(mat[i * nCols + j] != 0)
                    h_numNonZeros++;
            }
        }

        // Populate on host then copy to device
        uint32_t *h_rowPtrs = new uint32_t[nRows + 1];
        uint32_t *h_colIndx = new uint32_t[h_numNonZeros];
        float *h_value = new float[h_numNonZeros];

        size_t indx = 0;
        for(size_t i = 0; i < nRows; ++i){
            h_rowPtrs[i] = indx;
            for(size_t j = 0; j < nCols; ++j){
                if(mat[i * nCols + j] != 0){
                    h_colIndx[indx] = j;
                    h_value[indx] = mat[i * nCols + j];
                    indx++;
                }
            }
        }
        h_rowPtrs[nRows] = indx;
        
        // allocate memory
        CUDA_CHECK_ERROR(cudaMalloc(&d_numNonZeros, sizeof(size_t)));
        CUDA_CHECK_ERROR(cudaMalloc(&rowPtrs, (nRows + 1) * sizeof(uint32_t)));
        CUDA_CHECK_ERROR(cudaMalloc(&colIndx, h_numNonZeros * sizeof(uint32_t)));
        CUDA_CHECK_ERROR(cudaMalloc(&values, h_numNonZeros * sizeof(float)));

        // Copy to device
        CUDA_CHECK_ERROR(cudaMemcpy(rowPtrs, h_rowPtrs, (nRows + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK_ERROR(cudaMemcpy(colIndx, h_colIndx, h_numNonZeros * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK_ERROR(cudaMemcpy(values, h_value, h_numNonZeros * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK_ERROR(cudaMemcpy(d_numNonZeros, &h_numNonZeros, sizeof(size_t), cudaMemcpyHostToDevice));

        delete[] h_rowPtrs;
        delete[] h_colIndx;
        delete[] h_value;

        refCount = new std::atomic<int>(1);
    }

    // Copy constructor
    CSRMatrixGPU(const CSRMatrixGPU& other)
        : rowPtrs(other.rowPtrs), colIndx(other.colIndx), values(other.values),
          d_numNonZeros(other.d_numNonZeros), h_numNonZeros(other.h_numNonZeros),
          nRows(other.nRows), nCols(other.nCols), refCount(other.refCount) {
        // Increment the reference count
        if (refCount) {
            (*refCount)++;
        }
    }

    __device__ size_t getnRows() const {return nRows;}
    __device__ float getValues(uint32_t indx) {return values[indx];}
    __device__ float getRowPtrs(uint32_t indx) {return rowPtrs[indx];}
    __device__ float getColsIndx(uint32_t indx) {return colIndx[indx];}
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
    uint32_t *rowPtrs, *colIndx;
    std::atomic<int> *refCount = nullptr;
};