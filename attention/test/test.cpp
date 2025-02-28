#include <iostream>
#include "dot_product.cuh"
#include "utils.h"
#include <iomanip>

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

int main(){
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