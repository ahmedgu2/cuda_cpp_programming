#include <iostream>
#include <random>

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
    std::uniform_int_distribution<int> dist(0, 255);
    for(int i = 0; i < length; ++i){
        vector[i] = dist(gen);
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

void test_cpu_gpu(float sum_cpu_result, float sum_gpu_result){
    std::cout << "sum_cpu : " << sum_cpu_result << ", sum_gpu: " << sum_gpu_result << std::endl;
    if(sum_cpu_result != sum_gpu_result)
        std::cout << "\033[31mTEST FAILED\033[0m" << std::endl;
    else
        std::cout << "\033[32mTEST PASSED!\033[0m" << std::endl;    
}

__global__
void simpleSumReduction_gpu(float *input, const int length, float *output){
    /**
     * Simple sum kernel for input arrays that fit in 1 block.
     */
    const uint32_t indx = 2 * threadIdx.x;
    for(int stride = 1; stride <= blockDim.x; stride *= 2){
        if(threadIdx.x % stride == 0){
            input[indx] += input[indx + stride];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0){
        *output = input[0];
    }
}

__global__
void simpleSumReductionV2_gpu(float *input, const int length, float *output){
    /**
     * Optimizes `simpleSumReduction_gpu()` by decreasing control divergence. 
     * This is done by applying a better thread assignement strategy.
     */
    for(int stride = blockDim.x; stride >= 1; stride /= 2){
        if(threadIdx.x < stride){
            input[threadIdx.x] += input[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0){
        *output = input[0];
    }
}


int main(){
    const int length = 8;
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

    // Copy input data to GPU
    CUDA_CHECK_ERROR(cudaMemcpy(d_input, input, length * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    // Since we're using threadIdx.x * 2 in the kernel, we need length/2 threads
    int numThreads = length / 2;
    simpleSumReductionV2_gpu<<<1, numThreads>>>(d_input, length, d_sum_gpu_result);
    CUDA_KERNEL_CHECK_ERROR();

    // Copy result back to host
    CUDA_CHECK_ERROR(cudaMemcpy(&sum_gpu_result, d_sum_gpu_result, sizeof(float), cudaMemcpyDeviceToHost));

    // Compare results
    test_cpu_gpu(sum_cpu_result, sum_gpu_result);

    // Cleanup
    CUDA_CHECK_ERROR(cudaFree(d_input));
    CUDA_CHECK_ERROR(cudaFree(d_sum_gpu_result));
    delete[] input;
    
    return 0;
}
