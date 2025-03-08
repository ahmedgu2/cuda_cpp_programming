#include <cstdint>
#include <algorithm>
#include "utils.h"

void symmetricQuant_cpu(float *array, size_t length, int8_t *q_array){
    float abs_max = *std::max_element(
        array,
        array + length,
        [](float a, float b){
            return abs(a) < abs(b);
        }
    );
    int bits = 8;
    float S = abs(abs_max) / (powf(2, 8-1) - 1);
    for(int i = 0; i < length; ++i)
        q_array[i] = roundf(array[i] / S);
}

void asymmetricQuant_cpu(float *array, size_t length, uint8_t *q_array){
    float max_ = *std::max_element(array, array + length);
    float min_ = *std::min_element(array, array + length);
    float S = (max_ - min_) / 255;
    uint8_t Z = round(-min_ / S);
    for(int i = 0; i < length; ++i)
        q_array[i] = roundf(array[i] / S + Z);
}