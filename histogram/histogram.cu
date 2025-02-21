#include <cstdlib>
#include <iostream>

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

template<typename T>
void printArray(T *vector, const int length){
    for(int i = 0; i < length; ++i){
        std::cout << vector[i] << " ";
    }
    std::cout << std::endl;
}

void initArray(char *array, uint32_t length){
    for(int i = 0; i < length; ++i)
        array[i] = (std::rand() % 26) + 'a';
}

/********** Histogram kernel ************/
__global__
void histogram(char *array, uint32_t length, uint32_t *hist){
    uint32_t indx = threadIdx.x + blockDim.x * blockIdx.x;

    if(indx < length){
        int charVal = array[indx] - 'a';
        atomicAdd(&(hist[charVal / 4]), 1);
    }
}

int main(){
    char *array, *d_array;
    uint32_t *hist, *d_hist;
    uint32_t length = 10;
    
    array = new char[length];
    hist = new uint32_t[length];

    memset(hist, 0, length * sizeof(int));
    initArray(array, length);
    
    CUDA_CHECK_ERROR(cudaMalloc(&d_array, length * sizeof(char)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_hist, length * sizeof(uint32_t)));

    CUDA_CHECK_ERROR(cudaMemcpy(d_array, array, length * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemset(d_hist, 0, length * sizeof(uint32_t)));

    int THREAD_DIM = 512;
    int gridSize = (length + THREAD_DIM - 1) / THREAD_DIM;
    histogram<<<gridSize, THREAD_DIM>>>(d_array, length, d_hist);
    CUDA_KERNEL_CHECK_ERROR();

    CUDA_CHECK_ERROR(cudaMemcpy(hist, d_hist, length * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Print for testing
    printArray(hist, length);
    printArray(array, length);

    cudaFree(d_array);
    cudaFree(d_hist);
    delete[] array;
    delete[] hist;

    return 0;
}