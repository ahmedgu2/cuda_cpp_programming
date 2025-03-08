#pragma once
#include <cstdint>

void quantizeAsymmetric_gpu(float *array, size_t length, int bits, uint8_t *q_array);