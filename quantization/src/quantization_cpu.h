#pragma once
#include <cstdint>


void symmetricQuant_cpu(float *array, size_t length, int8_t *q_array);
void asymmetricQuant_cpu(float *array, size_t length, uint8_t *q_array);
void rowWiseQuant8bits_cpu(float *X, size_t nRows, size_t nCols, int8_t *q_X);