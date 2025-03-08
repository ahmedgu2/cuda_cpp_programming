#include "symmetric.cuh"
#include "asymmetric.cuh"
#include "quantization_cpu.h"
#include "utils.h"
#include <iostream>

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

void test_symmetricQuant(){
    int length = 1e5;
    try{
        float *array = new float[length];
        int8_t *q_array_gpu = new int8_t[length];
        int8_t *q_array_cpu = new int8_t[length];

        initVector(array, length);

        symmetricQuant_cpu(array, length, q_array_cpu);
        quantizeSymmetric_gpu(array, length, 8, q_array_gpu);

        compareArrays(q_array_gpu, q_array_cpu, length);

        delete[] array;
        delete[] q_array_cpu;
        delete[] q_array_gpu;

    }catch(std::bad_alloc& e){
        std::cerr << e.what();
    }
}

void test_asymmetricQuant(){
    int length = 1e5;
    try{
        float *array = new float[length];
        uint8_t *q_array_gpu = new uint8_t[length];
        uint8_t *q_array_cpu = new uint8_t[length];

        initVector(array, length);

        asymmetricQuant_cpu(array, length, q_array_cpu);
        quantizeAsymmetric_gpu(array, length, 8, q_array_gpu);

        compareArrays(q_array_gpu, q_array_cpu, length);

        delete[] array;
        delete[] q_array_cpu;
        delete[] q_array_gpu;

    }catch(std::bad_alloc& e){
        std::cerr << e.what();
    }
}

int main(){
    std::cout << "testing symmetric quantization" << std::endl;
    test_symmetricQuant();

    std::cout << "testing asymmetric quantization" << std::endl;
    test_asymmetricQuant();
}