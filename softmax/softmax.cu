#include <iomanip>
#include <algorithm>
#include <iostream>
#include <random>
#include <cmath>

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
void initVector(float *vector, const int length, unsigned int seed = 42){
    std::mt19937 gen(seed);
    // std::uniform_int_distribution<int> dist(1, 255);
    std::normal_distribution<float> dist;
    for(int i = 0; i < length; ++i){
        vector[i] = (float)dist(gen);
    }
}

void printVector(float *vector, const int length){
    for(int i = 0; i < length; ++i){
        std::cout << vector[i] << " ";
    }
    std::cout << std::endl;
}

float sum_cpu(float *input, const int length){
    float sum_ = 0;
    for(int i = 0; i < length; ++i){
        sum_ += input[i];
    }
    return sum_;
}

__global__
void maxArray_gpu(float *array, const int size, float *blocksMax){
    const int t = threadIdx.x;
    const int segmentDim = 2 * blockDim.x;
    const int indx = threadIdx.x + segmentDim * blockIdx.x;
    extern __shared__ float max_s[];

    if(indx < size)
        max_s[t] = array[indx];
    if(indx + blockDim.x < size)
        max_s[t] = fmax(max_s[t], array[indx + blockDim.x]);

    for(int stride = blockDim.x / 2; stride >= 1; stride >>= 1){
        __syncthreads();
        if(t < stride)
            max_s[t] = fmax(max_s[t], max_s[t + stride]);
    }
    if(t == 0)
        blocksMax[blockIdx.x] = max_s[0];
}

float max_gpu(float *array, const int size){
    constexpr int BLOCK_DIM = 1024;
    const int gridSize = (size + BLOCK_DIM - 1) / BLOCK_DIM;
    float *d_blocksMax;
    float *d_input;

    // Allocate GPU memory
    CUDA_CHECK_ERROR(cudaMalloc(&d_input, size * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_blocksMax, gridSize * sizeof(float)));

    // Copy input data to GPU
    CUDA_CHECK_ERROR(cudaMemcpy(d_input, array, size * sizeof(float), cudaMemcpyHostToDevice));

    maxArray_gpu<<<gridSize, BLOCK_DIM, BLOCK_DIM * sizeof(float)>>>(d_input, size, d_blocksMax);
    CUDA_KERNEL_CHECK_ERROR();
    // After first kernel launch, we get each block's max in `blocksMax` array
    // Now all we need is to launch a second kernel with only 1 block and `blocksMax` as input to get the global max from all blocks.
    // PS: No need to launch the second kernel if we only have 1 block.
    if(gridSize > 1){
        maxArray_gpu<<<1, BLOCK_DIM, gridSize * sizeof(float)>>>(d_blocksMax, gridSize, d_blocksMax);
        CUDA_KERNEL_CHECK_ERROR();
    }
    
    // Copy result to cpu
    float max_;
    CUDA_CHECK_ERROR(cudaMemcpy(&max_, &d_blocksMax[0], sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_blocksMax);
    cudaFree(d_input);
    
    return max_;
}


int main(){
    const int length = 510;
    float *input = new float[length];
    
    // Initialize input data
    initVector(input, length);

    // Compute CPU and GPU max
    std::cout << "max_cpu: " << *std::max_element(input, input + length) << std::endl;
    std::cout << "max_gpu: " << max_gpu(input, length);
    
    // Cleanup
    delete[] input;
    
    return 0;
}