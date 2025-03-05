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
#include "AttentionLayer.h"
#include "cuda_utils.cuh"

#define LOG(msg) std::cout << msg << std::endl

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

float* matmul_cpu(float *mat1, int nRows1, int nCols1, float *mat2, int nRows2, int nCols2){
    if(nCols1 != nRows2){
        std::cerr << "ERROR: nCols1 != nRows1. These must be equal to be able to apply matrix multiplcation!" << std::endl;
        exit(EXIT_FAILURE);
    }
    float *result = new float[nRows1 * nCols2];
    for(int row = 0; row < nRows1; ++row){
        for(int col = 0; col < nCols2; ++col){
            result[row * nCols2 + col] = 0.f;
            for(int k = 0; k < nCols1; ++k){
                result[row * nCols2 + col] += mat1[row * nCols1 + k] * mat2[k * nCols2 + col];
            }
        }
    }
    return result;
}

float* matmulTranspose_cpu(float *mat1, int nRows1, int nCols1, float *mat2, int nRows2, int nCols2){
    // if(nCols1 != nRows2){
    //     std::cerr << "ERROR: nCols1 != nRows2. These must be equal to be able to apply matrix multiplcation!" << std::endl;
    //     exit(EXIT_FAILURE);
    // }
    float *result = new float[nRows1 * nCols2];
    for(int row = 0; row < nRows1; ++row){
        for(int col = 0; col < nRows2; ++col){
            result[row * nRows2 + col] = 0.f;
            for(int k = 0; k < nCols1; ++k){
                result[row * nRows2 + col] += mat1[row * nCols1 + k] * mat2[col * nRows2 + k];
            }
        }
    }
    return result;
}

void divideByScalar(float *mat, int nRows, int nCols, float scalar){
    for(int row = 0; row < nRows; ++row){
        for(int col = 0; col < nCols; ++col){
            mat[row * nCols + col] /= scalar;
        }
    }
}

float *selfAttention_cpu(
    float *Q,
    float *K,
    float *V,
    float *W_q,
    float *W_k,
    float *W_v,
    const int seq_len,
    const int dim_emb
){
    float *QW = matmul_cpu(Q, seq_len, dim_emb, W_q, dim_emb, dim_emb);
    float *KW = matmul_cpu(K, seq_len, dim_emb, W_k, dim_emb, dim_emb);
    float *VW = matmul_cpu(V, seq_len, dim_emb, W_v, dim_emb, dim_emb);

    float *QK = matmulTranspose_cpu(QW, seq_len, dim_emb, KW, seq_len, dim_emb);
    float scalingFactor = sqrt(dim_emb);
    divideByScalar(QK, seq_len, seq_len, scalingFactor);
    auto QK_softmax_vec = softmax2D_cpu(QK, seq_len, seq_len);
    float *QK_softmax = QK_softmax_vec.data();
    float *attentionOutput = matmul_cpu(QK_softmax, seq_len, seq_len, VW, seq_len, dim_emb);

    delete[] QK;
    delete[] KW;
    delete[] VW;
    delete[] QW;
    // delete[] QK_softmax;

    return attentionOutput;
}

/************** Testing functions ************ */
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

void test_softmax2D(){
    const int nRows = 256;
    const int nCols = 256;
    float X[nRows * nCols];

    initVector(X, nRows * nCols);
    // printVector(X, nRows * nCols);
    auto softmaxResult_cpu = softmax2D_cpu(X, nRows, nCols);
    float *softmaxResult_gpu = softmax2D_gpu(X, nRows, nCols);
    // printMatrix(softmax_cpu, nRows, nCols);
    compareArrays(softmaxResult_gpu, softmaxResult_cpu.data(), nRows * nCols);

    delete[] softmaxResult_gpu;
}

void test_matmul(){
    const int nRows1 = 512, nRows2 = 128, nCols1 = 128, nCols2 = 512;
    float *X = new float[nRows1 * nCols1];
    float *Y = new float[nRows2 * nCols2];

    initVector(X, nRows1 * nCols1);
    initVector(Y, nRows2 * nCols2);

    float* matmulResult_cpu = matmul_cpu(X, nRows1, nCols1, Y, nRows2, nCols2);
    float* matmulResult_gpu = matmul_gpu(X, nRows1, nCols1, Y, nRows2, nCols2);
    compareArrays(matmulResult_gpu, matmulResult_cpu, nRows1 * nCols2);

    delete[] matmulResult_cpu;
    delete[] matmulResult_gpu;
    delete[] X;
    delete[] Y;
}

void test_attentionLayer(){
    uint32_t seq_len = 32, dim_emb = 256;
    AttentionLayer attentionLayer(seq_len, dim_emb, "cuda");
    // Create queries, keys and values
    float *queries = new float[seq_len * dim_emb];
    float *keys = new float[seq_len * dim_emb];
    float *values = new float[seq_len * dim_emb];

    // cpu version data (would be integrated later into AttentionLayer class)
    float *W_q = new float[seq_len * dim_emb];
    float *W_k = new float[seq_len * dim_emb];
    float *W_v = new float[seq_len * dim_emb];
    initArrayXavier(W_q, seq_len * dim_emb, dim_emb, dim_emb);
    initArrayXavier(W_k, seq_len * dim_emb, dim_emb, dim_emb);
    initArrayXavier(W_v, seq_len * dim_emb, dim_emb, dim_emb);

    // init 
    initVector(queries, seq_len * dim_emb);
    initVector(keys, seq_len * dim_emb);
    initVector(values, seq_len * dim_emb);

    // allocate device memory
    float *d_queries, *d_keys, *d_values;
    allocateCudaMemory(&d_queries, seq_len * dim_emb);
    allocateCudaMemory(&d_values, seq_len * dim_emb);
    allocateCudaMemory(&d_keys, seq_len * dim_emb);

    // Copy data
    copyToCuda(d_queries, queries, seq_len * dim_emb);
    copyToCuda(d_keys, keys, seq_len * dim_emb);
    copyToCuda(d_values, values, seq_len * dim_emb);
    
    float *attentionOutput_gpu = attentionLayer.forward(d_queries, d_keys, d_values);
    float *attentionOutput_cpu = selfAttention_cpu(queries, keys, values, W_q, W_k, W_v, seq_len, dim_emb);
    
    compareArrays(attentionOutput_gpu, attentionOutput_cpu, seq_len * dim_emb);

    delete[] attentionOutput_cpu;
    delete[] attentionOutput_gpu;
    freeCudaMemory(d_queries);
    freeCudaMemory(d_keys);
    freeCudaMemory(d_values);
    delete[] W_q;
    delete[] W_k;
    delete[] W_v;
}

int main(){
    std::cout << "Testing dotProduct..." << std::endl;
    test_dotProduct();
    
    std::cout << "Testing softmax2D..." << std::endl;
    test_softmax2D();

    std::cout << "Testing matmul..." << std::endl;
    test_matmul();

    std::cout << "Testing AttentionLayer..." << std::endl;
    test_attentionLayer();
}