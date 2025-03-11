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
 void rowWiseQuant8bits(float *X, size_t nRows, size_t nCols, int8_t *q_X, float *rowsMax){
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
    rowsMax[row] = max_s[0];
    float S = 127 / max_s[0];
    for(int i = tx; i < nCols; i += stride){
        int indx = row * nCols + i;
        q_X[indx] = __float2int_rn(S * X[indx]);
    }
}

__global__
void columnWiseQuant8bits(float *X, size_t nRows, size_t nCols, int8_t *q_X, float *columnsMax){
    /**
     * Apply quantization column-wise:
     *  Q_{X[i, j]} = round(S_j * X[i, j])
     * With:
     *  - S_j = 127 / max(abs(X[, j]))
     * 
     * Each block is responsible for getting the maximum value of a columns using parallel reduction, compute S_{col} and then quantize that column's values.
     */
    int tx = threadIdx.x;
    int stride = blockDim.x;
    int col = blockIdx.x;
    extern __shared__ float max_s[];

    // 1. Get column-wise max
    max_s[tx] = 0;
    for(int i = tx; i < nRows; i += stride){
        max_s[tx] = fmax(max_s[tx], abs(X[i * nCols + col]));
    }

    for(int i = stride / 2; i > 0; i >>= 1){
        __syncthreads();
        if(tx < i){
            max_s[tx] = fmax(max_s[tx], max_s[tx + i]);
        }
    }

    __syncthreads();
    // 2. Quantize column-wise
    columnsMax[col] = max_s[0];
    float S = 127 / max_s[0];
    for(int i = tx; i < nRows; i += stride){
        q_X[i * nCols + col] = __float2int_rn(S * X[i * nCols + col]);
    }
}

__global__
void matmul(int8_t *X1, size_t nRows1, size_t nCols1, int8_t *X2, size_t nRows2, size_t nCols2, int32_t *output){
    int tx = threadIdx.x, ty = threadIdx.y;
    int col = tx + blockDim.x * blockIdx.x;
    int row = ty + blockDim.y * blockIdx.y;
    const int TILE_WIDTH = blockDim.x;

    extern __shared__ int8_t shared_mem[];
    int8_t *X1_s = shared_mem;
    int8_t *X2_s = shared_mem + TILE_WIDTH * TILE_WIDTH;

    // 1. Load tiles into shared memory
    for(int tile = 0; tile < (nCols1 + TILE_WIDTH - 1) / TILE_WIDTH; ++tile){
        if(row < nRows1 && (col < (tile * TILE_WIDTH + tx)))
            X1_s[ty * TILE_WIDTH + tx] = X1[row * nCols1 + (tile * TILE_WIDTH + tx)];
        else
            X1_s[ty * TILE_WIDTH + tx] = 0;
        if((tile * TILE_WIDTH + ty) < nRows2 && col > nCols2)
            X2_s[ty * TILE_WIDTH + tx] = X2[(tile * TILE_WIDTH + ty) * nCols2 + col];
    }

}

void rowWiseQuant8bits_gpu(float *X, size_t nRows, size_t nCols, int8_t *q_X){
    const size_t length = nRows * nCols;
    float *d_X, *d_rowsMax;
    int8_t *d_q_X;

    CUDA_CHECK_ERROR(cudaMalloc(&d_X, length * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_q_X, length * sizeof(int8_t)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_rowsMax, nRows * sizeof(int8_t)));

    CUDA_CHECK_ERROR(cudaMemcpy(d_X, X, length * sizeof(float), cudaMemcpyHostToDevice));

    const int threadsPerBlock = 512;
    const int numBlocks = nRows;
    const size_t sharedMemorySize = threadsPerBlock * sizeof(float);
    rowWiseQuant8bits<<<numBlocks, threadsPerBlock, sharedMemorySize>>>(d_X, nRows, nCols, d_q_X, d_rowsMax);
    CUDA_KERNEL_CHECK_ERROR();

    CUDA_CHECK_ERROR(cudaMemcpy(q_X, d_q_X, length * sizeof(int8_t), cudaMemcpyDeviceToHost));

    CUDA_CHECK_ERROR(cudaFree(d_q_X));
    CUDA_CHECK_ERROR(cudaFree(d_X));
    CUDA_CHECK_ERROR(cudaFree(d_rowsMax));
}

void columnWiseQuant8bits_gpu(float *X, size_t nRows, size_t nCols, int8_t *q_X){
    const size_t length = nRows * nCols;
    float *d_X, *d_colsMax;
    int8_t *d_q_X;

    CUDA_CHECK_ERROR(cudaMalloc(&d_X, length * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_q_X, length * sizeof(int8_t)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_colsMax, nCols * sizeof(int8_t)));

    CUDA_CHECK_ERROR(cudaMemcpy(d_X, X, length * sizeof(float), cudaMemcpyHostToDevice));

    const int threadsPerBlock = 512;
    const int numBlocks = nCols;
    const size_t sharedMemorySize = threadsPerBlock * sizeof(float);
    columnWiseQuant8bits<<<numBlocks, threadsPerBlock, sharedMemorySize>>>(d_X, nRows, nCols, d_q_X, d_colsMax);
    CUDA_KERNEL_CHECK_ERROR();

    CUDA_CHECK_ERROR(cudaMemcpy(q_X, d_q_X, length * sizeof(int8_t), cudaMemcpyDeviceToHost));

    CUDA_CHECK_ERROR(cudaFree(d_q_X));
    CUDA_CHECK_ERROR(cudaFree(d_X));
    CUDA_CHECK_ERROR(cudaFree(d_colsMax));
}