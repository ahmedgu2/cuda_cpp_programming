#pragma once
#include <cstdint>

void quantizeSymmetric_gpu(float *array, size_t length, int bits, int8_t *q_array);