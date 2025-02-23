#include <iostream>
#include <random>
#include <iomanip>

/******************* Utility macros and functions ********************** */
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

/***************** GPU Code *********************** */

__global__
void multiplyVec(float *x, float *y, const int length, float *output){
    for(int indx = threadIdx.x + blockDim.x * blockIdx.x; indx < length; indx += gridDim.x * blockDim.x){
        output[indx] = x[indx] * y[indx];
    }
}

__global__
void sum(float *array, const int length, float *output){
    int t = threadIdx.x;
    int segment = 2 * blockDim.x;
    int indx = threadIdx.x + segment * blockIdx.x;
    extern __shared__ float sum_s[];

    if(indx < length)
        sum_s[t] = array[indx];
    if(indx + blockDim.x < length)
        sum_s[t] += array[indx + blockDim.x];

    for(int stride = blockDim.x / 2; stride >= 1; stride >>=1){
        __syncthreads();
        if(t < stride)
            sum_s[t] += sum_s[t + stride];
    }
    if(t == 0)
        atomicAdd(output, sum_s[0]);
}

/*********** Dot product calling function ************* */
float dot_gpu(float *x, float *y, const int length){
    float *d_x, *d_y, *d_mul, *d_dot;
    float dot = 0.f;
    size_t arraySize = length * sizeof(float);

    // Allocate device memory
    CUDA_CHECK_ERROR(cudaMalloc(&d_x, arraySize));
    CUDA_CHECK_ERROR(cudaMalloc(&d_y, arraySize));
    CUDA_CHECK_ERROR(cudaMalloc(&d_mul, arraySize));
    CUDA_CHECK_ERROR(cudaMalloc(&d_dot, sizeof(float)));
    
    // Init
    CUDA_CHECK_ERROR(cudaMemcpy(d_x, x, arraySize, cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_y, y, arraySize, cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemset(d_dot, 0, sizeof(float)));

    // Kernels call
    constexpr int BLOCK_DIM = 512;
    const int gridSize = (length + BLOCK_DIM - 1) / BLOCK_DIM;
    const int sharedMemorySize = BLOCK_DIM * sizeof(float);
    multiplyVec<<<gridSize, BLOCK_DIM>>>(d_x, d_y, length, d_mul);
    CUDA_KERNEL_CHECK_ERROR();
    sum<<<gridSize, BLOCK_DIM, sharedMemorySize>>>(d_mul, length, d_dot);
    CUDA_KERNEL_CHECK_ERROR();

    // Copy output to host
    CUDA_CHECK_ERROR(cudaMemcpy(&dot, d_dot, sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_dot);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_mul);

    return dot;
}

/*************** CPU Version ***************** */
float dot_cpu(float *x, float *y, const int length){
    float dotResult = 0.f;
    for(int i = 0; i < length; ++i){
        dotResult += x[i] * y[i];
    }
    return dotResult;
}

int main(){

    const int length = 1024;
    float x[length], y[length];

    initVector(x, length);
    initVector(y, length);

    float dotResult_gpu = dot_gpu(x, y, length);
    float dotResult_cpu = dot_cpu(x, y, length);
    std::cout << std::fixed << std::setprecision(7) << dotResult_cpu << " " << dotResult_gpu << std::endl;
    if(std::abs(dotResult_cpu - dotResult_gpu) > 1e-3){
        std::cerr << "\033[31mTEST FAILED\033[0m" << std::endl;
        exit(EXIT_FAILURE);
    }
    std::cout << "\033[32mTEST PASSED!\033[0m" << std::endl;

    return 0;
}