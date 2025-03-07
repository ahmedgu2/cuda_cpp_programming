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

void printMatrix(std::vector<float>& mat, const int nRows, const int nCols);
void initArrayXavier(float *array, const int length, const int fan_in, const int fan_out, unsigned int seed = 42);