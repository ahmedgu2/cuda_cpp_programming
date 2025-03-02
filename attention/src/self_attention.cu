#include "cuda_macros.cuh"
#include "utils.h"

__global__
void softmax2D(float *X, const int nRows, const int nCols, float *softmaxResult){
    /**
     * @brief Implement softmax 2D row-wise:
     * softmax(X_{i,j}) = exp(X_{i, j} - max(X_i)) / sum^{nCols}_{i=0} exp(X_{i, j} - max(X_i))
     * 
     */
    int row = blockIdx.x;
    int tx = threadIdx.x;
    int stride = blockDim.x;
    int indx = row * nCols;

    extern __shared__ float shared_mem[];
    float *max_s = (float *)shared_mem;
    float *sum_s = (float *)shared_mem + stride;

    // 1. Calculate the maximum for row
    max_s[tx] = - INFINITY;
    for(int i = tx; i < nCols; i += stride){
        max_s[tx] = fmax(max_s[tx], X[indx + i]);
    }
    
    for(int i = stride >> 1; i >= 1; i >>= 1){
        __syncthreads();
        if(tx < i)
            max_s[tx] = fmax(max_s[tx], max_s[tx + i]);
    }

    __syncthreads();

    // 2. Calculate the sum 
    float maxVal = max_s[0];
    sum_s[tx] = 0.f;
    for(int i = tx; i < nCols; i += stride){
        sum_s[tx] += __expf(X[indx + i] - maxVal);
    }
    
    for(int i = stride >> 1; i >= 1; i >>= 1){
        __syncthreads();
        if(tx < i)
            sum_s[tx] += sum_s[tx + i];
    }

    __syncthreads();

    // 3. Normalize
    float sumVal = sum_s[0] + 1e-5; // Add epsilon for numerical stability
    for(int i = tx; i < nCols; i += stride){
        softmaxResult[indx + i] = __expf(X[indx + i] - maxVal) / sumVal; 
    }
}

float* softmax2D_gpu(float *X, const int nRows, const int nCols){

    float *d_X, *d_softmax;
    size_t size = nRows * nCols * sizeof(float);
    CUDA_CHECK_ERROR(cudaMalloc(&d_X, size));
    CUDA_CHECK_ERROR(cudaMalloc(&d_softmax, size));

    CUDA_CHECK_ERROR(cudaMemcpy(d_X, X, size, cudaMemcpyHostToDevice));

    dim3 BLOCK_DIM(512);
    dim3 gridSize(nRows);
    size_t sharedMemorySize = 2 * BLOCK_DIM.x * sizeof(float);
    softmax2D<<<gridSize, BLOCK_DIM, sharedMemorySize>>>(d_X, nRows, nCols, d_softmax);
    CUDA_KERNEL_CHECK_ERROR();

    float *softmax_host = new float[nRows * nCols];
    CUDA_CHECK_ERROR(cudaMemcpy(softmax_host, d_softmax, size, cudaMemcpyDeviceToHost));
    return softmax_host;
}