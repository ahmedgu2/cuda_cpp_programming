#include "symmetric.cuh"
#include "utils.h"
#include <iostream>

int main(){
    float array[8] = {1.2, -0.5, -4.3, 1.2, -3.1, 0.8, 2.4, 5.4};
    int8_t q_array[8];
    int length = 8;
    quantize_gpu(array, length, 8, q_array); // Expected output: 28 -12 -101 28 -73 19 56 127

    for(int i = 0; i < length; ++i)
        printf("%d ", q_array[i]); 
}