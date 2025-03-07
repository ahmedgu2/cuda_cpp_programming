#include "cuda_macros.cuh"
#include "utils.h"
#include "symmetric.cuh"

__global__
void max_(float *array, size_t length, float* blockMax){
    extern __shared__ float max_s[];
    int tx = threadIdx.x; 
    int segment = 2 * blockDim.x; 
    int indx = tx + segment * blockIdx.x;

    max_s[tx] = -INFINITY;
    if(indx < length)
        max_s[tx] = abs(array[indx]);
    if(indx + blockDim.x < length)
        max_s[tx] = fmax(max_s[tx], abs(array[indx + blockDim.x]));

    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1){
        __syncthreads();
        if(tx < stride)
            max_s[tx] = fmax(max_s[tx], max_s[tx + stride]);
    }
    if(tx == 0){
        blockMax[blockIdx.x] = max_s[tx];
    }
}

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

void quantize_gpu(float *array, size_t length, int bits, int8_t *q_array){
    float *d_array, *d_blockMax, *d_max;
    int8_t *d_q_array;
    
    CUDA_CHECK_ERROR(cudaMalloc(&d_array, length * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_q_array, length * sizeof(uint8_t)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_blockMax, length * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_max, sizeof(float)));
    
    CUDA_CHECK_ERROR(cudaMemcpy(d_array, array, length * sizeof(float), cudaMemcpyHostToDevice));

    // Run max kernel to get partial max for each block
    int threadsPerBlock = 512;
    int numBlocks = (length + threadsPerBlock - 1) / threadsPerBlock;
    int sharedMemorySize = threadsPerBlock * sizeof(float);
    max_<<<numBlocks, threadsPerBlock, sharedMemorySize>>>(d_array, length, d_blockMax);
    CUDA_KERNEL_CHECK_ERROR();
    cudaDeviceSynchronize();

    // Call max_ a second time to reduce block-wise maximums
    float max;
    if(numBlocks > 1){ // No need to call if the first call had only 1 block.
        sharedMemorySize = numBlocks * sizeof(float);
        // Assumes we only need 1 pass to get the final result (Can be generalized)
        max_<<<1, numBlocks, sharedMemorySize>>>(d_blockMax, numBlocks, d_max);
        CUDA_KERNEL_CHECK_ERROR();
        cudaDeviceSynchronize();
        CUDA_CHECK_ERROR(cudaMemcpy(&max, d_max, sizeof(float), cudaMemcpyDeviceToHost));
    }else{
        CUDA_CHECK_ERROR(cudaMemcpy(&max, d_blockMax, sizeof(float), cudaMemcpyDeviceToHost));
    }

    float S = max / (powf(2, bits - 1) - 1);

    threadsPerBlock = 1024;
    quantize<<<numBlocks, threadsPerBlock>>>(d_array, length, S, d_q_array);
    CUDA_KERNEL_CHECK_ERROR();
    
    CUDA_CHECK_ERROR(cudaMemcpy(q_array, d_q_array, length * sizeof(int8_t), cudaMemcpyDeviceToHost));
}