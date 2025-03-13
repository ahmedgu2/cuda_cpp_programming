#include <iostream>
#include <random>

// Utils
void initVector(float *vector, const int length, unsigned int seed = 42)
{
    std::mt19937 gen(seed);
    // std::uniform_int_distribution<int> dist(1, 255);
    std::normal_distribution<float> dist;
    for (int i = 0; i < length; ++i){
        vector[i] = (float)dist(gen);
    }
}

void initVectorInt(float *vector, const int length, unsigned int seed = 42)
{
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(1, 8);
    // std::normal_distribution<float> dist;
    for (int i = 0; i < length; ++i){
        vector[i] = (float)dist(gen);
    }
}

void printMatrix(float* mat, const int nRows, const int nCols){
    for(int row = 0; row < nRows; ++row){
        for(int col = 0; col < nCols; ++col){
            int indx = row * nCols + col;
            std::cout << mat[indx] << " ";
        }
        std::cout << std::endl;
    }
}

std::normal_distribution<float> xavierNormalDist(int fan_in, int fan_out){
    float stddev = sqrt(2.0 / (fan_in + fan_out));
    return std::normal_distribution<float>(0.0, stddev);
}

void initArrayXavier(float *array, const int length, const int fan_in, const int fan_out, unsigned int seed = 42){
    std::mt19937 gen(seed);
    auto dist = xavierNormalDist(fan_in, fan_out);
    for (int i = 0; i < length; ++i){
        array[i] = (float)dist(gen);
    }
}

void matmul_cpu(float *mat1, int nRows1, int nCols1, float *mat2, int nRows2, int nCols2, float *result){
    if(nCols1 != nRows2){
        std::cerr << "ERROR: nCols1 != nRows1. These must be equal to be able to apply matrix multiplcation!" << std::endl;
        exit(EXIT_FAILURE);
    }
    for(int row = 0; row < nRows1; ++row){
        for(int col = 0; col < nCols2; ++col){
            result[row * nCols2 + col] = 0.f;
            for(int k = 0; k < nCols1; ++k){
                result[row * nCols2 + col] += mat1[row * nCols1 + k] * mat2[k * nCols2 + col];
            }
        }
    }
}