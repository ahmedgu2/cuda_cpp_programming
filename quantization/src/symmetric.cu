#include "cuda_macros.cuh"
#include "utils.h"
#include "symmetric.cuh"
#include "cuda_utils.cuh"

__global__
void quantize(float *array, size_t length, float S, int8_t *q_array){
    int stride = gridDim.x * blockDim.x;
    for(int indx = threadIdx.x + blockDim.x * blockIdx.x; indx < length; indx += stride){
        q_array[indx] = __float2int_rn(array[indx] / S); // Converts float to nearest int
    }
}

__global__
void dequantize(int8_t *q_array, size_t length, float S, float *dq_array){
    int stride = gridDim.x * blockDim.x;
    for(int indx = threadIdx.x + blockDim.x * blockIdx.x; indx < length; indx += stride){
        dq_array[indx] = q_array[indx] * S;
    }
}

void quantizeSymmetric_gpu(float *array, size_t length, int bits, int8_t *q_array){
    float *d_array, *d_blockMax;
    int8_t *d_q_array;
    int threadsPerBlock = 512;
    int numBlocks = (length + threadsPerBlock - 1) / threadsPerBlock;
    
    CUDA_CHECK_ERROR(cudaMalloc(&d_array, length * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_q_array, length * sizeof(int8_t)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_blockMax, numBlocks * sizeof(float)));
    
    CUDA_CHECK_ERROR(cudaMemcpy(d_array, array, length * sizeof(float), cudaMemcpyHostToDevice));

    // Run max kernel to get partial max for each block
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

    float S = max / (powf(2, bits - 1) - 1);
    quantize<<<numBlocks, threadsPerBlock>>>(d_array, length, S, d_q_array);
    CUDA_KERNEL_CHECK_ERROR();
    
    CUDA_CHECK_ERROR(cudaMemcpy(q_array, d_q_array, length * sizeof(int8_t), cudaMemcpyDeviceToHost));

    CUDA_CHECK_ERROR(cudaFree(d_array));
    CUDA_CHECK_ERROR(cudaFree(d_q_array));
    CUDA_CHECK_ERROR(cudaFree(d_blockMax));
}