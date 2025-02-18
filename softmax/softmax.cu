#include <iomanip>
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
        max_s[t] = fmax(array[indx], array[indx + blockDim.x]);

    for(int stride = blockDim.x / 2; stride >= 1; stride >= 1){
        __syncthreads();
        if(t < stride)
            max_s[t] = fmax(max_s[t], max_s[t + stride]);
    }
    if(t == 0)
        blocksMax[blockIdx.x] = max_s[0];
}

float max_gpu(float *array, const int size){
    constexpr int BLOCK_DIM = 512;
    const int gridSize = (size + BLOCK_DIM - 1) / BLOCK_DIM;
    float *blocksMax = new float[gridSize];

    maxArray_gpu<<<gridSize, BLOCK_DIM, BLOCK_DIM>>>(array, size, blocksMax);
    maxArray_gpu<<<1, BLOCK_DIM, BLOCK_DIM>>>(array, size, blocksMax);

    // Copy result to cpu
    float max_;
    CUDA_CHECK_ERROR(cudaMemcpy(&max_, &blocksMax[0], sizeof(float), cudaMemcpyDeviceToHost));

    delete[] blocksMax;
    
    return max_;
}


int main(){
    const int length = 1024 * 1024;
    float *input = new float[length];
    float sum_cpu_result, sum_gpu_result;
    float *d_input, *d_sum_gpu_result;
    
    // Initialize input data
    initVector(input, length);

    // Compute CPU sum
    sum_cpu_result = sum_cpu(input, length);

    // Allocate GPU memory
    CUDA_CHECK_ERROR(cudaMalloc(&d_input, length * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_sum_gpu_result, sizeof(float)));
    // Before launching the kernel, add:
    CUDA_CHECK_ERROR(cudaMemset(d_sum_gpu_result, 0.f, sizeof(float)));

    // Copy input data to GPU
    CUDA_CHECK_ERROR(cudaMemcpy(d_input, input, length * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    int numThreads = 256;
    int numBlocks = (length + numThreads - 1) / numThreads;
    size_t sharedMemorySize = numThreads * sizeof(float);
    // sumReductionSharedMultiSegment_gpu<<<numBlocks, numThreads, sharedMemorySize>>>(d_input, length, d_sum_gpu_result);
    CUDA_KERNEL_CHECK_ERROR();

    // Copy result back to host
    CUDA_CHECK_ERROR(cudaMemcpy(&sum_gpu_result, d_sum_gpu_result, sizeof(float), cudaMemcpyDeviceToHost));

    // Compare results
    // test_cpu_gpu(sum_cpu_result, sum_gpu_result);

    // Cleanup
    CUDA_CHECK_ERROR(cudaFree(d_input));
    CUDA_CHECK_ERROR(cudaFree(d_sum_gpu_result));
    delete[] input;
    
    return 0;
}