#include <torch/extension.h>
#include <iostream>
#include "symmetric.cuh"

torch::Tensor symmetric_quantization_gpu(torch::Tensor& array){
    auto q_array = torch::zeros_like(array, torch::kInt8); // Output tensor
    
    // Get raw data
    float *array_ptr = array.data_ptr<float>();
    float *q_array_ptr = q_array.data_ptr<int8_t>();

    size_t length = array.numel();
    quantizeSymmetric_gpu(array_ptr, length, 8, q_array_ptr);

    return q_array;
}