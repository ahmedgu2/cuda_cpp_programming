#include "cuda_macros.cuh"
#include <cstdint>

/**
 * TODO: implement the following:
 *  - Vector-wise quantization using absmax quantization (symmetric one).
 *  - Outlier / non-outlier seperations.
 *  - Apply mixed-precision matmul (outliers in fp16, non-outlier in int8).
 *  - Think about how to kernel fuse the quantization and matmul together for optimal performance.
 *  - Benchmark with cpu and pytorch cuda implementation (I need to implement both)
 */

 __global__
 void rowWiseQuant8bits(float *X, size_t nRows, size_t nCols, int8_t *q_X){
    /**
     * Apply quantization row-wise:
     *  Q_{X[i, j]} = round(S_i * X[i, j])
     * With:
     *  - S_i = 127 / max(abs(X[i,]))
     * 
     * Each block is responsible for getting the maximum value of a row using parallel reduction, compute S_{row} and then quantize that row's values.
     */

    int tx = threadIdx.x;
    int stride = blockDim.x;
    int row = blockIdx.x;
    extern __shared__ float max_s[];
     
    // 1. Get row-wise maxabs
    // Load in shared memory and reduce first level
    max_s[tx] = 0;
    for(int i = tx; i < nCols; i += stride){
        max_s[tx] = fmax(abs(X[row * nCols + i]), max_s[tx]);
    }
    // Reduce the rest
    for(int i = stride / 2; i > 0; i >>= 1){
        __syncthreads();
        if(tx < i)
            max_s[tx] = fmax(max_s[tx], max_s[tx + i]);
    }
    __syncthreads();

    // 2. Quantize row
    float S = 127 / max_s[0];
    for(int i = tx; i < nCols; i += stride){
        int indx = row * nCols + i;
        q_X[indx] = __float2int_rn(S * X[indx]);
    }
}

void rowWiseQuant8bits_gpu(float *X, size_t nRows, size_t nCols, int8_t *q_X){
    const size_t length = nRows * nCols;
    float *d_X;
    int8_t *d_q_X;

    CUDA_CHECK_ERROR(cudaMalloc(&d_X, length * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_q_X, length * sizeof(int8_t)));

    CUDA_CHECK_ERROR(cudaMemcpy(d_X, X, length * sizeof(float), cudaMemcpyHostToDevice));

    const int threadsPerBlock = 512;
    const int numBlocks = nRows;
    const size_t sharedMemorySize = threadsPerBlock * sizeof(float);
    rowWiseQuant8bits<<<numBlocks, threadsPerBlock, sharedMemorySize>>>(d_X, nRows, nCols, d_q_X);
    CUDA_KERNEL_CHECK_ERROR();

    CUDA_CHECK_ERROR(cudaMemcpy(q_X, d_q_X, length * sizeof(int8_t), cudaMemcpyDeviceToHost));

    CUDA_CHECK_ERROR(cudaFree(d_q_X));
    CUDA_CHECK_ERROR(cudaFree(d_X));
}