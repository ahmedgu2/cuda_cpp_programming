#include <iomanip>
#include <numeric>
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
void maxArray_gpu(float *array, const int length, float *blocksMax){
    const int t = threadIdx.x;
    const int segmentDim = 2 * blockDim.x;
    const int indx = threadIdx.x + segmentDim * blockIdx.x;
    extern __shared__ float max_s[];

    if(indx < length)
        max_s[t] = array[indx];
    if(indx + blockDim.x < length)
        max_s[t] = fmax(max_s[t], array[indx + blockDim.x]);

    for(int stride = blockDim.x / 2; stride >= 1; stride >>= 1){
        __syncthreads();
        if(t < stride)
            max_s[t] = fmax(max_s[t], max_s[t + stride]);
    }
    if(t == 0)
        blocksMax[blockIdx.x] = max_s[0];
}

__global__
void expArray_gpu(float *array, const int length, float *output){
    for(int indx = threadIdx.x + blockDim.x * blockIdx.x; indx < length; indx += gridDim.x * blockDim.x){
        output[indx] = expf(array[indx]);
    }
}

__global__
void sumArray_gpu(float *array, const int length, float *output){
    const int t = threadIdx.x;
    const int segment = 2 * blockDim.x;
    const int indx = threadIdx.x + segment * blockIdx.x;
    extern __shared__ float sum_s[];

    sum_s[t] = 0.f;
    if(indx < length)
        sum_s[t] = array[indx];
    if(indx + blockDim.x < length)
        sum_s[t] += array[indx + blockDim.x];

    for(int stride = blockDim.x / 2; stride >= 1; stride >>= 1){
        __syncthreads();
        if(t < stride)
            sum_s[t] += sum_s[t + stride];
    }
    if(t == 0)
        atomicAdd(output, sum_s[0]);
}

__global__
void substractArray_gpu(float *array1, float val, float *output, const int length){
    for(int indx = threadIdx.x + blockDim.x * blockIdx.x; indx < length; indx += gridDim.x * blockDim.x){
        output[indx] = array1[indx] - val;
    }
}

__global__
void divideArray_gpu(float *array1, float val, float *output, const int length){
    for(int indx = threadIdx.x + blockDim.x * blockIdx.x; indx < length; indx += gridDim.x * blockDim.x){
        output[indx] = array1[indx] / val;
    }
}

float max_gpu(float *array, const int size){
    constexpr int BLOCK_DIM = 512;
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

float* softmax_gpu(float *array, const int length){
    float *d_array, *d_output, *d_sum;
    const int size = length * sizeof(float);
    const int BLOCK_DIM = 512;
    int gridSize = (length + BLOCK_DIM - 1) / BLOCK_DIM;

    // Allocate gpu memory
    CUDA_CHECK_ERROR(cudaMalloc(&d_array, size));
    CUDA_CHECK_ERROR(cudaMalloc(&d_output, size));
    CUDA_CHECK_ERROR(cudaMalloc(&d_sum, sizeof(float)));
    // Init d_sum
    CUDA_CHECK_ERROR(cudaMemset(d_sum, 0.f, sizeof(float)));

    // Copy host array to device
    CUDA_CHECK_ERROR(cudaMemcpy(d_array, array, size, cudaMemcpyHostToDevice));

    // Softmax calculation:
    // 1. Compute max(array). softmax(x + c) = softmax(x). We'll set c = -max(array) for numerical stability (avoid overflow)
    float max_ = max_gpu(array, length);

    // 2. Substract max from array
    substractArray_gpu<<<gridSize, BLOCK_DIM>>>(d_array, max_, d_output, length);
    CUDA_KERNEL_CHECK_ERROR();

    // 3. Calculate exp(x - c)
    expArray_gpu<<<gridSize, BLOCK_DIM>>>(d_output, length, d_output);
    CUDA_KERNEL_CHECK_ERROR();

    // 4. Calculate the normalization term (sum (exp(x - c)))
    const int sharedMemorySize = BLOCK_DIM * sizeof(float);
    sumArray_gpu<<<gridSize, BLOCK_DIM, sharedMemorySize>>>(d_output, length, d_sum);
    CUDA_KERNEL_CHECK_ERROR();
    float sum_;
    CUDA_CHECK_ERROR(cudaMemcpy(&sum_, d_sum, sizeof(float), cudaMemcpyDeviceToHost));

    // 5. Get final result by dividing exp(x - c) by the normalization term + epsilon (to avoid deviding by 0)
    const float epsilon = 1e-5;
    divideArray_gpu<<<gridSize, BLOCK_DIM>>>(d_output, sum_ + epsilon, d_output, length);
    CUDA_KERNEL_CHECK_ERROR();

    // Copy result to host memory
    float *softmaxResult = new float[length];
    CUDA_CHECK_ERROR(cudaMemcpy(softmaxResult, d_output, size, cudaMemcpyDeviceToHost));

    cudaFree(d_array);
    cudaFree(d_output);
    cudaFree(d_sum);

    return softmaxResult;
}

/***************************** Kernel fusion ******************************* */

__global__
void sumExpArray_gpu(float *array, const int length, float *output, float *exp_x, float max_){
    /**
     * This kernel computes and returns exp(x-max) and the normalization term (sum(exp(x - c)))
     */
    const int t = threadIdx.x;
    const int segment = 2 * blockDim.x;
    const int indx = threadIdx.x + segment * blockIdx.x;
    extern __shared__ float sum_s[];

    sum_s[t] = 0.f;
    if(indx < length){
        float val = expf(array[indx] - max_);
        sum_s[t] = val;
        exp_x[indx] = val;
    }
    if(indx + blockDim.x < length){
        float val = expf(array[indx + blockDim.x] - max_);
        sum_s[t] += val;
        exp_x[indx + blockDim.x] = val;
    }

    for(int stride = blockDim.x / 2; stride >= 1; stride >>= 1){
        __syncthreads();
        if(t < stride)
            sum_s[t] += sum_s[t + stride];
    }
    if(t == 0)
        atomicAdd(output, sum_s[0]);
}

float* softmaxFused_gpu(float *array, const int length){
    float *d_array, *d_output, *d_sum, *d_exp_x;
    const int size = length * sizeof(float);
    const int BLOCK_DIM = 512;
    int gridSize = (length + BLOCK_DIM - 1) / BLOCK_DIM;

    // Allocate gpu memory
    CUDA_CHECK_ERROR(cudaMalloc(&d_array, size));
    CUDA_CHECK_ERROR(cudaMalloc(&d_output, size));
    CUDA_CHECK_ERROR(cudaMalloc(&d_sum, sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_exp_x, size));
    // Init d_sum
    CUDA_CHECK_ERROR(cudaMemset(d_sum, 0.f, sizeof(float)));

    // Copy host array to device
    CUDA_CHECK_ERROR(cudaMemcpy(d_array, array, size, cudaMemcpyHostToDevice));

    // Softmax calculation:
    // 1. Compute max(array). softmax(x + c) = softmax(x). We'll set c = -max(array) for numerical stability (avoid overflow)
    float max_ = max_gpu(array, length);

    // 2. Calculate the exp(x-c) and normalization term (sum (exp(x - c)))
    const int sharedMemorySize = BLOCK_DIM * sizeof(float);
    sumExpArray_gpu<<<gridSize, BLOCK_DIM, sharedMemorySize>>>(d_array, length, d_sum, d_exp_x, max_);
    CUDA_KERNEL_CHECK_ERROR();
    float sum_;
    CUDA_CHECK_ERROR(cudaMemcpy(&sum_, d_sum, sizeof(float), cudaMemcpyDeviceToHost));

    // 3. Get final result by dividing exp(x - c) by the normalization term + epsilon (to avoid deviding by 0)
    const float epsilon = 1e-5;
    divideArray_gpu<<<gridSize, BLOCK_DIM>>>(d_exp_x, sum_ + epsilon, d_output, length);
    CUDA_KERNEL_CHECK_ERROR();

    // Copy result to host memory
    float *softmaxResult = new float[length];
    CUDA_CHECK_ERROR(cudaMemcpy(softmaxResult, d_output, size, cudaMemcpyDeviceToHost));

    cudaFree(d_array);
    cudaFree(d_output);
    cudaFree(d_sum);

    return softmaxResult;
}

/* Test kernels*/
void compareKernelsOutputs(float *softmaxResults, float *softmaxFusedResults, const int length, const float eps=1e4){
    for(int i = 0; i < length; ++i){
        if(std::abs(softmaxFusedResults[i] - softmaxResults[i]) > eps){
            std::cerr << "Mismatch at index " << i << ": " << softmaxResults[i] << " != " << softmaxFusedResults[i] << std::endl;
            std::cerr << "\033[31mTEST FAILED\033[0m" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    std::cout << "\033[32mTEST PASSED!\033[0m" << std::endl;
}

int main(){
    const int length = 4096*1024;
    float *input = new float[length];
    
    // Initialize input data
    initVector(input, length);
    
    // Test softmax
    float *softmaxResult = softmax_gpu(input, length);
    // float *softmaxFusedResult = softmaxFused_gpu(input, length);

    // compareKernelsOutputs(softmaxResult, softmaxFusedResult, length);
    
    std::cout << "################# softmax: #####################\n";

    // Cleanup
    delete[] softmaxResult;
    // delete[] softmaxFusedResult;
    delete[] input;
    
    return 0;
}