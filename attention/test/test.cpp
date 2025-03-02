#include <iostream>
#include <cmath>
#include "dot_product.cuh"
#include "utils.h"
#include "self_attention.cuh"
#include <iomanip>
#include <algorithm>
#include <vector>
#include <numeric>
#include "self_attention.cuh"

/*************** CPU Version ***************** */
float dot_cpu(float *x, float *y, const int length)
{
    float dotResult = 0.f;
    for (int i = 0; i < length; ++i)
    {
        dotResult += x[i] * y[i];
    }
    return dotResult;
}

std::vector<float> softmax2D_cpu(float *X, const int nRows, const int nCols){
    std::vector<float> maxs(nRows);
    std::vector<float> sums(nRows);
    for(int row = 0; row < nRows; ++row){
        float *row_start = X + row * nCols;
        maxs[row] = *std::max_element(row_start, row_start + nCols);
        sums[row] = std::accumulate(row_start, row_start + nCols, 0.f,
            [&maxs, row](float accumulator, float element){
                return accumulator + std::exp(element - maxs[row]);
            }
        );
    }

    std::vector<float> result(nRows * nCols);
    for(int row = 0; row < nRows; ++row){
        for(int col = 0; col < nCols; ++col){
            int idx = row * nCols + col;
            result[idx] = std::exp(X[idx] - maxs[row]) / sums[row];
        }
    }
    return result;
}

/************** Testing functions ************ */
void test_dotProduct(){
    const int length = 1024;
    float x[length], y[length];

    initVector(x, length);
    initVector(y, length);

    float dotResult_gpu = dot_gpu(x, y, length);
    float dotResult_cpu = dot_cpu(x, y, length);
    std::cout << std::fixed << std::setprecision(7) << dotResult_cpu << " " << dotResult_gpu << std::endl;
    if(std::abs(dotResult_cpu - dotResult_gpu) > 1e-3){
        std::cerr << "\033[31mTEST FAILED\033[0m" << std::endl;
        exit(EXIT_FAILURE);
    }
    std::cout << "\033[32mTEST PASSED!\033[0m" << std::endl;
}

void compareArrays(float *x, float *y, const int length){
    for(int i = 0; i < length; ++i){
        if(std::abs(x[i] - y[i]) > 1e-3){
            std::cout << "Mismatch at index : " << i << " " << x[i] << " != " << y[i] << std::endl;
            std::cerr << "\033[31mTEST FAILED\033[0m" << std::endl;
            // exit(EXIT_FAILURE);
        }
    }
    std::cout << "\033[32mTEST PASSED!\033[0m" << std::endl;
}

void test_softmax2D(){
    const int nRows = 2;
    const int nCols = 3;
    float X[nRows * nCols];

    initVector(X, nRows * nCols);
    printVector(X, nRows * nCols);
    auto softmaxResult_cpu = softmax2D_cpu(X, nRows, nCols);
    float *softmaxResult_gpu = softmax2D_gpu(X, nRows, nCols);
    // printMatrix(softmax_cpu, nRows, nCols);
    compareArrays(softmaxResult_gpu, softmaxResult_cpu.data(), nRows * nCols);

    delete[] softmaxResult_gpu;
}

int main(){
    std::cout << "Testing dotProduct..." << std::endl;
    test_dotProduct();
    
    std::cout << "Testing softmax2D..." << std::endl;
    test_softmax2D();
}