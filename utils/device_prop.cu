#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

#define CUDA_CHECK_ERROR(callResult) do{ \
    cudaError_t error = callResult; \
    if(error != cudaSuccess){ \
        std::cout << "Error in " << __FILE__ << " at line " << __LINE__ << "\n" << cudaGetErrorString(error); \
        exit(EXIT_FAILURE); \
    } \
}while(0)

int main() {
    int deviceCount;
    CUDA_CHECK_ERROR(cudaGetDeviceCount(&deviceCount));

    for (int device = 0; device < deviceCount; device++) {
        cudaDeviceProp deviceProp;
        CUDA_CHECK_ERROR(cudaGetDeviceProperties(&deviceProp, device));

        printf("\nDevice %d: %s\n", device, deviceProp.name);
        printf("Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("Total global memory: %.2f GB\n", deviceProp.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        
        // SM Properties
        printf("Number of SMs: %d\n", deviceProp.multiProcessorCount);
        printf("Max blocks per SM: %d\n", deviceProp.maxBlocksPerMultiProcessor);
        printf("Max threads per SM: %d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("Max shared memory per SM: %.2f KB\n", deviceProp.sharedMemPerMultiprocessor / 1024.0);
        printf("Max registers per SM: %d\n", deviceProp.regsPerMultiprocessor);
        
        // Thread and Block Properties
        printf("Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
        printf("Max threads dimensions: (%d, %d, %d)\n", 
            deviceProp.maxThreadsDim[0],
            deviceProp.maxThreadsDim[1],
            deviceProp.maxThreadsDim[2]);
        printf("Max grid dimensions: (%d, %d, %d)\n",
            deviceProp.maxGridSize[0],
            deviceProp.maxGridSize[1],
            deviceProp.maxGridSize[2]);
        printf("Warp size: %d\n", deviceProp.warpSize);
    }

    return 0;
}