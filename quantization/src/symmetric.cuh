#pragma once
#include <cstdint>

void quantize_gpu(float *array, size_t length, int bits, int8_t *q_array);