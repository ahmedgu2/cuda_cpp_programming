#pragma once
#include <vector>
#include <iostream>

void initVector(float *vector, const int length, unsigned int seed = 42);

template <typename T>
void printVector(T *vector, const int length){
    for (int i = 0; i < length; ++i){
        std::cout << vector[i] << " ";
    }
    std::cout << std::endl;
}

void printMatrix(float* mat, const int nRows, const int nCols);
void initArrayXavier(float *array, const int length, const int fan_in, const int fan_out, unsigned int seed = 42);
void matmul_cpu(float *mat1, int nRows1, int nCols1, float *mat2, int nRows2, int nCols2, float *result);
void initVectorInt(float *vector, const int length, unsigned int seed = 42);