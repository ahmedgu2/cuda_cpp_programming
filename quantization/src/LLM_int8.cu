#include "cuda_macros.cuh"
#include "LLM_int8.cuh"
#include <cstdint>
#include <cuda_fp16.h>

/**
 * TODO: implement the following:
 *  - Vector-wise quantization using absmax quantization (symmetric one).
 *  - Outlier / non-outlier seperations.
 *  - Apply mixed-precision matmul (outliers in fp16, non-outlier in int8).
 *  - Think about how to kernel fuse the quantization and matmul together for optimal performance.
 *  - Benchmark with cpu and pytorch cuda implementation (I need to implement both)
 */

 __global__
 void rowWiseQuant8bits(__half *X, size_t nRows, size_t nCols, int8_t *q_X, __half *rowsScale){
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
    extern __shared__ __half max_s[];
     
    // 1. Get row-wise maxabs
    // Load in shared memory and reduce first level
    max_s[tx] = __float2half(0.f);
    for(int i = tx; i < nCols; i += stride){
        max_s[tx] = __hmax(__habs(X[row * nCols + i]), max_s[tx]);
    }
    // Reduce the rest
    for(int i = stride / 2; i > 0; i >>= 1){
        __syncthreads();
        if(tx < i)
            max_s[tx] = __hmax(max_s[tx], max_s[tx + i]);
    }
    __syncthreads();

    // 2. Quantize row
    __half S = __hdiv(__float2half(127.f), max_s[0]);
    rowsScale[row] = S;
    for(int i = tx; i < nCols; i += stride){
        int indx = row * nCols + i;
        q_X[indx] = __half2int_rn(__hmul(S, X[indx]));
    }
}

__global__
void columnWiseQuant8bits(float *X, size_t nRows, size_t nCols, int8_t *q_X, float *columnsScale){
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
    extern __shared__ float max_sc[];

    // 1. Get column-wise max
    max_sc[tx] = 0;
    for(int i = tx; i < nRows; i += stride){
        max_sc[tx] = fmax(max_sc[tx], abs(X[i * nCols + col]));
    }

    for(int i = stride / 2; i > 0; i >>= 1){
        __syncthreads();
        if(tx < i){
            max_sc[tx] = fmax(max_sc[tx], max_sc[tx + i]);
        }
    }

    __syncthreads();
    // 2. Quantize column-wise
    float S = 127 / max_sc[0];
    columnsScale[col] = S;
    for(int i = tx; i < nRows; i += stride){
        q_X[i * nCols + col] = __float2int_rn(S * X[i * nCols + col]);
    }
}

template <typename IN_TYPE, typename OUT_TYPE>
__global__
void matmul(IN_TYPE *X1, size_t nRows1, size_t nCols1, IN_TYPE *X2, size_t nRows2, size_t nCols2, OUT_TYPE *output){
    int tx = threadIdx.x, ty = threadIdx.y;
    int col = tx + blockDim.x * blockIdx.x;
    int row = ty + blockDim.y * blockIdx.y;
    const int TILE_WIDTH = blockDim.x;

    extern __shared__ __align__(sizeof(IN_TYPE)) unsigned char shared_mem[];
    IN_TYPE *X1_s = reinterpret_cast<IN_TYPE*>(shared_mem);
    IN_TYPE *X2_s = X1_s + TILE_WIDTH * TILE_WIDTH;

    // 1. Load tiles into shared memory
    OUT_TYPE val = 0;
    for(int tile = 0; tile < (nCols1 + TILE_WIDTH - 1) / TILE_WIDTH; ++tile){
        if(row < nRows1 && ( (tile * TILE_WIDTH + tx) < nCols1))
            X1_s[ty * TILE_WIDTH + tx] = X1[row * nCols1 + (tile * TILE_WIDTH + tx)];
        else
            X1_s[ty * TILE_WIDTH + tx] = 0;
        if((tile * TILE_WIDTH + ty) < nRows2 && col < nCols2)
            X2_s[ty * TILE_WIDTH + tx] = X2[(tile * TILE_WIDTH + ty) * nCols2 + col];
        else
            X2_s[ty * TILE_WIDTH + tx] = 0;
        __syncthreads();

        for(int k = 0; k < TILE_WIDTH; ++k)
            val += X1_s[ty * TILE_WIDTH + k] * X2_s[k * TILE_WIDTH + tx];

        __syncthreads();
    }
    if(row < nRows1 && col < nCols2)
        output[row * nCols2 + col] = val;
}

__global__
void dequentize(int32_t *q_X, size_t nRows, size_t nCols, float *rowsScale, float *columnsScale, float *X){
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int x = threadIdx.x + blockDim.x * blockIdx.x;

    // Grid-stride loop
    int stride_x = blockDim.x * gridDim.x;
    int stride_y = blockDim.y * gridDim.y;
    for(int row = y; row < nRows; row += stride_x){
        for(int col = x; col < nCols; col += stride_y){
            X[row * nCols + col] = q_X[row * nCols + col] * (1 / (rowsScale[row] * columnsScale[col]));
        }
    }
}

__global__
void outliersColumns(float* X, size_t nRows, size_t nCols, bool *isOutlierCol, float threshold = 6.f){
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int x = threadIdx.x + blockDim.x * blockIdx.x;

    // Grid-stride loop
    int stride_x = blockDim.x * gridDim.x;
    int stride_y = blockDim.y * gridDim.y;
    for(int row = y; row < nRows; row += stride_x){
        for(int col = x; col < nCols; col += stride_y){
            if(X[row * nCols + col] >= threshold)
                isOutlierCol[col] = 1;
        }
    }
}

/************************************************* Kernel wrappers ******************************************************** */
void rowWiseQuant8bits_gpu(__half *X, size_t nRows, size_t nCols, int8_t *q_X){
    const size_t length = nRows * nCols;
    __half *d_X, *d_rowsMax;
    int8_t *d_q_X;

    CUDA_CHECK_ERROR(cudaMalloc(&d_X, length * sizeof(__half)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_q_X, length * sizeof(int8_t)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_rowsMax, nRows * sizeof(int8_t)));

    CUDA_CHECK_ERROR(cudaMemcpy(d_X, X, length * sizeof(__half), cudaMemcpyHostToDevice));

    const int threadsPerBlock = 1024;
    const int numBlocks = nRows;
    const size_t sharedMemorySize = threadsPerBlock * sizeof(__half);
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

template <typename IN_TYPE, typename OUT_TYPE>
void matmul_gpu(IN_TYPE *X1, size_t nRows1, size_t nCols1, IN_TYPE *X2, size_t nRows2, size_t nCols2, OUT_TYPE *result){
    size_t size1 = nRows1 * nCols1 * sizeof(IN_TYPE);
    size_t size2 = nRows2 * nCols2 * sizeof(IN_TYPE);
    size_t sizer = nRows1 * nCols2 * sizeof(OUT_TYPE);

    IN_TYPE *d_mat1, *d_mat2;
    OUT_TYPE *d_result;
    CUDA_CHECK_ERROR(cudaMalloc(&d_mat1, size1));
    CUDA_CHECK_ERROR(cudaMalloc(&d_mat2, size2));
    CUDA_CHECK_ERROR(cudaMalloc(&d_result, sizer));

    // Copy matrices to device
    CUDA_CHECK_ERROR(cudaMemcpy(d_mat1, X1, size1, cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_mat2, X2, size2, cudaMemcpyHostToDevice));

    // Launch Kernel
    const int TILE_WIDTH = 16;
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH); // number of threads per dimension needs to be equal to TILE_DIM.
    dim3 numBlocks((nCols2 + threadsPerBlock.y - 1) / threadsPerBlock.y, (nRows1 + threadsPerBlock.x - 1) / threadsPerBlock.x);
    size_t sharedMemorySize = 2 * TILE_WIDTH * TILE_WIDTH * sizeof(IN_TYPE);

    matmul<IN_TYPE, OUT_TYPE><<<numBlocks, threadsPerBlock, sharedMemorySize>>>(d_mat1, nRows1, nCols1, d_mat2, nRows2, nCols2, d_result);
    CUDA_KERNEL_CHECK_ERROR();

    // Copy to host
    CUDA_CHECK_ERROR(cudaMemcpy(result, d_result, sizer, cudaMemcpyDeviceToHost));
    
    cudaFree(d_mat1);
    cudaFree(d_mat2);
    cudaFree(d_result);
}

void outliersColumns_gpu(float *X, size_t nRows, size_t nCols, bool *isOutlierCol, float threshold){
    const size_t length = nRows * nCols;
    float *d_X;
    bool *d_isOutlierCol;

    CUDA_CHECK_ERROR(cudaMalloc(&d_X, length * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_isOutlierCol, nCols * sizeof(bool)));

    CUDA_CHECK_ERROR(cudaMemcpy(d_X, X, length * sizeof(float), cudaMemcpyHostToDevice));

    const int BLOCK_DIM = 32;
    dim3 threadsPerBlock(BLOCK_DIM, BLOCK_DIM); // number of threads per dimension needs to be equal to TILE_DIM.
    dim3 numBlocks((nCols + threadsPerBlock.y - 1) / threadsPerBlock.y, (nRows + threadsPerBlock.x - 1) / threadsPerBlock.x);

    outliersColumns<<<numBlocks, threadsPerBlock>>>(d_X, nRows, nCols, d_isOutlierCol);
    CUDA_KERNEL_CHECK_ERROR();

    CUDA_CHECK_ERROR(cudaMemcpy(isOutlierCol, d_isOutlierCol, nCols * sizeof(bool), cudaMemcpyDeviceToHost));

    CUDA_CHECK_ERROR(cudaFree(d_X));
    CUDA_CHECK_ERROR(cudaFree(d_isOutlierCol));
}


template void matmul_gpu(int8_t*, size_t, size_t, int8_t*, size_t, size_t, int32_t*);
template void matmul_gpu(float*, size_t, size_t, float*, size_t, size_t, float*);