#include <iostream>
#include <cuda_runtime.h>
#include <random>

/**************************************************************************************
 * Utils functions
 */

 #define CUDA_CHECK_ERROR(callResult) do{ \
    cudaError_t error = callResult; \
    if(error != cudaSuccess){ \
        std::cout << "Error in " << __FILE__ << " at line " << __LINE__ << "\n" << cudaGetErrorString(error); \
        exit(EXIT_FAILURE); \
    } \
}while(0)

#define CUDA_KERNEL_CHECK_ERROR() do{ \
    cudaError_t error = cudaGetLastError(); \
    if(error != cudaSuccess){ \
        std::cerr << "----CUDA ERROR in " << __FILE__ << " at line " << __LINE__ << "\n" << cudaGetErrorString(error) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}while(0)

// Utils
void initMatrix(float *matrix, int nRows, int nCols, unsigned int seed = 42){
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(0, 255);
    for(int i = 0; i < nRows; ++i){
        for(int j = 0; j < nCols; ++j){
            matrix[i * nCols + j] = dist(gen);
        }
    }
}

void printMatrix(float *matrix, int nRows, int nCols){
    for(int i = 0; i < nRows; ++i){
        for(int j = 0; j < nCols; ++j){
            std::cout << matrix[i * nRows + j] << " ";
        }
        std::cout << std::endl;
    }
}
/************************************************************************************* */

// Activation functions for 2D arrays
__global__
void sigmoid(float *mat, float *result, const int nRows, const int nCols){
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int indx = row * nCols + col;

    if(col < nCols && row < nRows){
        result[indx] = 1 / (1 + expf(-mat[indx]));
    }
}

__global__
void relu(float *mat, float *result, const int nRows, const int nCols){
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int indx = row * nCols + col;

    if(col < nCols && row < nRows){
        result[indx] = max(0.f, mat[indx]);
    }
}

__global__
void tanh(float *mat, float *result, const int nRows, const int nCols){
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int indx = row * nCols + col;

    if(col < nCols && row < nRows){
        result[indx] = tanhf(mat[indx]);
    }
}


int main(){
    int nRows = 2056, nCols = 1024;
    size_t size = nRows * nCols * sizeof(float);
    float *mat = new float[nRows * nCols];
    float *d_mat, *d_result;

    initMatrix(mat, nRows, nCols);

    // Allocate cuda memory
    CUDA_CHECK_ERROR(cudaMalloc(&d_mat, size));
    CUDA_CHECK_ERROR(cudaMalloc(&d_result, size));

    // Init device arrays
    CUDA_CHECK_ERROR(cudaMemcpy(d_mat, mat, size, cudaMemcpyHostToDevice));

    // Run kernels
    const int BLOCK_SIZE = 16;
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((nRows + BLOCK_SIZE - 1) / BLOCK_SIZE, (nCols + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // 1. Sigmoid
    sigmoid<<<numBlocks, threadsPerBlock>>>(d_mat, d_result, nRows, nCols);
    cudaDeviceSynchronize();
    CUDA_KERNEL_CHECK_ERROR();
    // Test
    // compareWithCpu()

    // 2. Relu
    sigmoid<<<numBlocks, threadsPerBlock>>>(d_mat, d_result, nRows, nCols);
    cudaDeviceSynchronize();
    CUDA_KERNEL_CHECK_ERROR();
    // Test TODO: add comparision with cpu functions output
    // compareWithCpu()
    
    // 2. Tanh
    sigmoid<<<numBlocks, threadsPerBlock>>>(d_mat, d_result, nRows, nCols);
    cudaDeviceSynchronize();
    CUDA_KERNEL_CHECK_ERROR();
    // Test
    // compareWithCpu()

    delete[] mat;
    cudaFree(d_mat);
    cudaFree(d_result);

    return 0;
}