#include "cuda_macros.cuh"
#include "utils.h"
#include "symmetric.cuh"
#include "cuda_utils.cuh"

__global__
void quantize(float *array, size_t length, float S, uint8_t Z, uint8_t *q_array){
    int stride = gridDim.x * blockDim.x;
    for(int indx = threadIdx.x + blockDim.x * blockIdx.x; indx < length; indx += stride){
        q_array[indx] = __float2int_rn(array[indx] / S + Z); // Converts float to nearest int
    }
}

__global__
void dequantize(uint8_t *q_array, size_t length, float S, uint8_t Z, float *dq_array){
    int stride = gridDim.x * blockDim.x;
    for(int indx = threadIdx.x + blockDim.x * blockIdx.x; indx < length; indx += stride){
        dq_array[indx] = (q_array[indx] - Z) * S;
    }
}

void quantizeAsymmetric_gpu(float *array, size_t length, int bits, uint8_t *q_array){
    float *d_array, *d_blockMax;
    uint8_t *d_q_array;
    int threadsPerBlock = 512;
    int numBlocks = (length + threadsPerBlock - 1) / threadsPerBlock;
    
    CUDA_CHECK_ERROR(cudaMalloc(&d_array, length * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_q_array, length * sizeof(uint8_t)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_blockMax, numBlocks * sizeof(float)));
    
    CUDA_CHECK_ERROR(cudaMemcpy(d_array, array, length * sizeof(float), cudaMemcpyHostToDevice));

    // Run MAX kernel to get partial max for each block
    int sharedMemorySize = threadsPerBlock * sizeof(float);
    max_<<<numBlocks, threadsPerBlock, sharedMemorySize>>>(d_array, length, d_blockMax);
    CUDA_KERNEL_CHECK_ERROR();
    cudaDeviceSynchronize();
    // Call max_ a second time to reduce block-wise maximums
    if(numBlocks > 1){ // No need to call if the first call had only 1 block.
        sharedMemorySize = numBlocks * sizeof(float);
        // Assumes we only need 1 pass to get the final result (Can be generalized)
        max_<<<1, numBlocks, sharedMemorySize>>>(d_blockMax, numBlocks, d_blockMax);
        CUDA_KERNEL_CHECK_ERROR();
        cudaDeviceSynchronize();
    }
    float max;
    CUDA_CHECK_ERROR(cudaMemcpy(&max, &d_blockMax[0], sizeof(float), cudaMemcpyDeviceToHost));

    // Run MIN kernel to get partial max for each block
    sharedMemorySize = threadsPerBlock * sizeof(float);
    // Re-use d_blockMax for min as well to avoid unecessary array allocation for min.
    min_<<<numBlocks, threadsPerBlock, sharedMemorySize>>>(d_array, length, d_blockMax);
    CUDA_KERNEL_CHECK_ERROR();
    cudaDeviceSynchronize();
    // Call max_ a second time to reduce block-wise maximums
    if(numBlocks > 1){ // No need to call if the first call had only 1 block.
        sharedMemorySize = numBlocks * sizeof(float);
        // Assumes we only need 1 pass to get the final result (Can be generalized)
        min_<<<1, numBlocks, sharedMemorySize>>>(d_blockMax, numBlocks, d_blockMax);
        CUDA_KERNEL_CHECK_ERROR();
        cudaDeviceSynchronize();
    }
    float min;
    CUDA_CHECK_ERROR(cudaMemcpy(&min, &d_blockMax[0], sizeof(float), cudaMemcpyDeviceToHost));

    float S = (max - min) / 255;
    uint8_t Z = roundf(-min / S);
    quantize<<<numBlocks, threadsPerBlock>>>(d_array, length, S, Z, d_q_array);
    CUDA_KERNEL_CHECK_ERROR();
    
    CUDA_CHECK_ERROR(cudaMemcpy(q_array, d_q_array, length * sizeof(uint8_t), cudaMemcpyDeviceToHost));

    CUDA_CHECK_ERROR(cudaFree(d_array));
    CUDA_CHECK_ERROR(cudaFree(d_q_array));
    CUDA_CHECK_ERROR(cudaFree(d_blockMax));
}