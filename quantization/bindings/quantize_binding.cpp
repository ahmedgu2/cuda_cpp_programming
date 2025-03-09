#include <torch/extension.h>
#include <iostream>
#include "symmetric.cuh"
#include <pybind11/pybind11.h>
namespace py = pybind11;

torch::Tensor symmetric_quantization_gpu(torch::Tensor& array){
    auto q_array = torch::zeros_like(array, torch::kInt8); // Output tensor
    
    // Get raw data
    float *array_ptr = array.data_ptr<float>();
    int8_t *q_array_ptr = q_array.data_ptr<int8_t>();

    size_t length = array.numel();
    quantizeSymmetric_gpu(array_ptr, length, 8, q_array_ptr);

    return q_array;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("symmetric_quantization_gpu", &symmetric_quantization_gpu, 
          "Symmetric quantization on GPU", py::arg("array"));
}