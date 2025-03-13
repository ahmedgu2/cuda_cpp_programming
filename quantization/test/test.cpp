#include "symmetric.cuh"
#include "asymmetric.cuh"
#include "quantization_cpu.h"
#include "LLM_int8.cuh"
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

void test_rowWiseQuant8bits(){
    const size_t nRows = 5, nCols = 1024;
    size_t length = nRows * nCols;

    try{
        float *X = new float[length];
        int8_t *q_X_cpu = new int8_t[length];
        int8_t *q_X_gpu = new int8_t[length];

        initVector(X, length);

        rowWiseQuant8bits_cpu(X, nRows, nCols, q_X_cpu);
        rowWiseQuant8bits_gpu(X, nRows, nCols, q_X_gpu);
        
        compareArrays(q_X_gpu, q_X_cpu, length);

        delete[] X;
        delete[] q_X_cpu;
        delete[] q_X_gpu;

    }catch(std::bad_alloc& e){
        std::cerr << e.what();
    }
}

void test_columnWiseQuant8bits(){
    const size_t nRows = 1024, nCols = 5;
    size_t length = nRows * nCols;

    try{
        float *X = new float[length];
        int8_t *q_X_cpu = new int8_t[length];
        int8_t *q_X_gpu = new int8_t[length];

        initVector(X, length);

        columnWiseQuant8bits_cpu(X, nRows, nCols, q_X_cpu);
        columnWiseQuant8bits_gpu(X, nRows, nCols, q_X_gpu);
        
        compareArrays(q_X_gpu, q_X_cpu, length);

        delete[] X;
        delete[] q_X_cpu;
        delete[] q_X_gpu;

    }catch(std::bad_alloc& e){
        std::cerr << e.what();
    }
}

void test_matmul(){
    const size_t nRows1 = 512, nCols1 = 1024;
    const size_t nRows2 = 1024, nCols2 = 256;
    size_t length1 = nRows1 * nCols1;
    size_t length2 = nRows2 * nCols2;
    size_t lengthr = nRows1 * nCols2;

    try{
        float *X = new float[length1];
        float *Y = new float[length2];
        float *result_gpu = new float[lengthr];
        float *result_cpu = new float[lengthr];

        initVector(X, length1);
        initVector(Y, length2);

        matmul_gpu<float, float>(X, nRows1, nCols1, Y, nRows2, nCols2, result_gpu);
        matmul_cpu(X, nRows1, nCols1, Y, nRows2, nCols2, result_cpu);
        
        compareArrays(result_gpu, result_cpu, lengthr);

        delete[] X;
        delete[] Y;
        delete[] result_cpu;
        delete[] result_gpu;

    }catch(std::bad_alloc& e){
        std::cerr << e.what();
    }
}

void test_outliersColumns(){
    const size_t nRows = 512, nCols = 1024;
    size_t length = nRows * nCols;

    try{
        float *X = new float[length];
        bool *isOutlierCol_cpu = new bool[nCols];
        bool *isOutlierCol_gpu = new bool[nCols];

        memset(isOutlierCol_cpu, 0, nCols * sizeof(bool));
        memset(isOutlierCol_gpu, 0, nCols * sizeof(bool));

        initVectorInt(X, length);

        outliersColumns_cpu(X, nRows, nCols, isOutlierCol_cpu);
        outliersColumns_gpu(X, nRows, nCols, isOutlierCol_gpu);
        compareArrays(isOutlierCol_gpu, isOutlierCol_cpu, nCols);

        delete[] X;
        delete[] isOutlierCol_cpu;
        delete[] isOutlierCol_gpu;

    }catch(std::bad_alloc& e){
        std::cerr << e.what();
    }
}

int main(){
    std::cout << "testing symmetric quantization" << std::endl;
    test_symmetricQuant();

    std::cout << "testing asymmetric quantization" << std::endl;
    test_asymmetricQuant();

    std::cout << "testing 8 bits row-wise quantization..." << std::endl;
    test_rowWiseQuant8bits();

    std::cout << "testing 8 bits column-wise quantization..." << std::endl;
    test_columnWiseQuant8bits();

    std::cout << "testing matmul" << std::endl;
    test_matmul();

    std::cout << "testing outliers detection..." << std::endl;
    test_outliersColumns();
}