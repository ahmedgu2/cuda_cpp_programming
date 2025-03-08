#pragma once
#include <cstdint>

__global__ void max_(float *array, size_t length, float* blockMax);
__global__ void min_(float *array, size_t length, float* blockMax);