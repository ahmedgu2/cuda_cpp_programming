#pragma once
#include <iostream>
#include <vector>
#include "cuda_macros.cuh"
#include <cstdint>

class COOMatrixGPU{
public:

    COOMatrixGPU(float* mat, size_t nRows, size_t nCols){
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
        uint32_t *h_rowIndx = new uint32_t[h_numNonZeros];
        uint32_t *h_colIndx = new uint32_t[h_numNonZeros];
        float *h_value = new float[h_numNonZeros];

        size_t indx = 0;
        for(size_t i = 0; i < nRows; ++i){
            for(size_t j = 0; j < nCols; ++j){
                if(mat[i * nCols + j] != 0){
                    h_rowIndx[indx] = i;
                    h_colIndx[indx] = j;
                    h_value[indx] = mat[i * nCols + j];
                    indx++;
                }
            }
        }
        
        // allocate memory
        CUDA_CHECK_ERROR(cudaMalloc(&numNonZeros, sizeof(size_t)));
        CUDA_CHECK_ERROR(cudaMalloc(&rowIndx, h_numNonZeros * sizeof(uint32_t)));
        CUDA_CHECK_ERROR(cudaMalloc(&colIndx, h_numNonZeros * sizeof(uint32_t)));
        CUDA_CHECK_ERROR(cudaMalloc(&value, h_numNonZeros * sizeof(float)));

        // Copy to device
        CUDA_CHECK_ERROR(cudaMemcpy(rowIndx, h_rowIndx, h_numNonZeros * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK_ERROR(cudaMemcpy(colIndx, h_colIndx, h_numNonZeros * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK_ERROR(cudaMemcpy(value, h_value, h_numNonZeros * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK_ERROR(cudaMemcpy(numNonZeros, &h_numNonZeros, sizeof(size_t), cudaMemcpyHostToDevice));

        delete[] h_rowIndx;
        delete[] h_colIndx;
        delete[] h_value;
    }

    ~COOMatrixGPU(){
        CUDA_CHECK_ERROR(cudaFree(rowIndx));
        CUDA_CHECK_ERROR(cudaFree(colIndx));
        CUDA_CHECK_ERROR(cudaFree(value));
        CUDA_CHECK_ERROR(cudaFree(numNonZeros));
    }

    __device__ uint32_t getRowIndx(uint32_t i) const {return rowIndx[i];}
    __device__ uint32_t getColIndx(uint32_t i) const {return colIndx[i];}
    __device__ float getValue(uint32_t i) const {return value[i];}
    __host__ __device__ size_t getNumNonZeros() const {
        #ifdef __CUDA_ARCH__
            // Safe to dereference device pointer
            return *numNonZeros;
        #else
            // Fallback to host pointer
            return h_numNonZeros;
        #endif
    }

private:
    uint32_t* rowIndx, *colIndx;
    float *value;
    size_t *numNonZeros;
    size_t h_numNonZeros;
    size_t nRows, nCols;
};