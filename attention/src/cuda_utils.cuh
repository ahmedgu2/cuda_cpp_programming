#pragma once
#include <cuda_runtime.h>

void allocateCudaMemory(float **ptr, size_t size);
void freeCudaMemory(float *ptr);