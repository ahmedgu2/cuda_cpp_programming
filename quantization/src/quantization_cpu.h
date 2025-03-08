#pragma once
#include <cstdint>


void symmetricQuant_cpu(float *array, size_t length, int8_t *q_array);
void asymmetricQuant_cpu(float *array, size_t length, uint8_t *q_array);