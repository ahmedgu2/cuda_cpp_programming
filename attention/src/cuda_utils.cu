#include "cuda_macros.cuh"

void allocateCudaMemory(float **ptr, size_t size){
    CUDA_CHECK_ERROR(cudaMalloc(ptr, size * sizeof(float)));
}

void freeCudaMemory(float *ptr){
    CUDA_CHECK_ERROR(cudaFree(ptr));
}

void copyToCuda(float *dst_ptr, float *src_ptr, size_t size){
    CUDA_CHECK_ERROR(cudaMemcpy(dst_ptr, src_ptr, size * sizeof(float), cudaMemcpyHostToDevice));
}