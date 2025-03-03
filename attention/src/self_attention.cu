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

__global__
void matmul(
    float *mat1, 
    const int nRows1, 
    const int nCols1, 
    float *mat2, 
    const int nRows2, 
    const int nCols2,
    float* output, 
    const size_t TILE_WIDTH
){
    uint32_t col = threadIdx.x + blockIdx.x * TILE_WIDTH;
    uint32_t row = threadIdx.y + blockIdx.y * TILE_WIDTH; 
    int tx = threadIdx.x, ty = threadIdx.y;

    extern __shared__ float shared_mem[];
    float *mat1_s = (float*)shared_mem;
    float *mat2_s = (float*)shared_mem + TILE_WIDTH * TILE_WIDTH;

    // Load tile into shared memory
    float outputVal = 0.f;

    for(int tile = 0; tile < (nCols1 + TILE_WIDTH - 1) / TILE_WIDTH; ++tile){
        uint32_t indx_s  = tx + TILE_WIDTH * ty;
        
        if(row < nRows1 && (TILE_WIDTH * tile + tx < nCols1))
            mat1_s[indx_s] = mat1[row * nCols1 + TILE_WIDTH * tile + tx];
        else
            mat1_s[indx_s] = 0.f;
        if((TILE_WIDTH * tile + ty < nRows2) && col < nCols2)
            mat2_s[indx_s] = mat2[(TILE_WIDTH * tile + ty) * nCols2 + col];
        else
            mat2_s[indx_s] = 0.f;

        __syncthreads();

        // Compute dot product 
        for(int k = 0; k < TILE_WIDTH; ++k){
            outputVal += mat1_s[TILE_WIDTH * ty + k] * mat2_s[TILE_WIDTH * k + tx];
        }
        __syncthreads();

    }
    if(row < nRows1 && col < nCols2)
        output[row * nCols2 + col] = outputVal;
}

float* matmul_gpu(
    float *mat1,
    const int nRows1,
    const int nCols1,
    float *mat2,
    const int nRows2,
    const int nCols2
){

    size_t size1 = nRows1 * nCols1 * sizeof(float);
    size_t size2 = nRows2 * nCols2 * sizeof(float);
    size_t sizer = nRows1 * nCols2 * sizeof(float);

    float *d_mat1, *d_mat2, *d_result;
    CUDA_CHECK_ERROR(cudaMalloc(&d_mat1, size1));
    CUDA_CHECK_ERROR(cudaMalloc(&d_mat2, size2));
    CUDA_CHECK_ERROR(cudaMalloc(&d_result, sizer));

    std::cout << "aaaaaa";
    // Copy matrices to device
    CUDA_CHECK_ERROR(cudaMemcpy(d_mat1, mat1, size1, cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_mat2, mat2, size2, cudaMemcpyHostToDevice));

    std::cout << "bbbb";
    // Launch Kernel
    const int TILE_WIDTH = 16;
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH); // number of threads per dimension needs to be equal to TILE_DIM.
    dim3 numBlocks((nCols2 + threadsPerBlock.y - 1) / threadsPerBlock.y, (nRows1 + threadsPerBlock.x - 1) / threadsPerBlock.x);
    size_t sharedMemorySize = 2 * TILE_WIDTH * TILE_WIDTH * sizeof(float);

    std::cout << "cccc";
    matmul<<<numBlocks, threadsPerBlock, sharedMemorySize>>>(d_mat1, nRows1, nCols1, d_mat2, nRows2, nCols2, d_result, TILE_WIDTH);
    CUDA_KERNEL_CHECK_ERROR();

    // Copy to host
    float *result_host = new float[sizer];
    CUDA_CHECK_ERROR(cudaMemcpy(result_host, d_result, sizer, cudaMemcpyDeviceToHost));
    
    cudaFree(d_mat1);
    cudaFree(d_mat2);
    cudaFree(d_result);

    return result_host;
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