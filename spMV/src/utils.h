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

template <typename T>
void compareArrays(T *x, T *y, const int length){
    for(int i = 0; i < length; ++i){
        if(std::abs(x[i] - y[i]) > 1e-3){
            std::cout << "Mismatch at index : " << i << " " << static_cast<int>(x[i]) << " != " << static_cast<int>(y[i]) << std::endl;
            std::cerr << "\033[31mTEST FAILED\033[0m" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    std::cout << "\033[32mTEST PASSED!\033[0m" << std::endl;
}

void printMatrix(float* mat, const int nRows, const int nCols);
void initArrayXavier(float *array, const int length, const int fan_in, const int fan_out, unsigned int seed = 42);
void matmul_cpu(float *mat1, int nRows1, int nCols1, float *mat2, int nRows2, int nCols2, float *result);
void initVectorInt(float *vector, const int length, unsigned int seed = 42);