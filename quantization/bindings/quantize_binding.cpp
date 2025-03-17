#include <torch/extension.h>
#include <iostream>
#include "symmetric.cuh"
#include "LLM_int8.cuh"
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

// torch::Tensor row_wise_quantization_gpu(torch::Tensor& tensor){
//     auto q_tensor = torch::zeros_like(tensor, torch::kInt8);

//     int nRows = tensor.size(0);
//     int nCols = tensor.size(1);

//     __half *tensor_ptr = reinterpret_cast<__half*>(tensor.data_ptr<at::Half>());
//     int8_t *q_tensor_ptr = q_tensor.data_ptr<int8_t>();

//     rowWiseQuant8bits_gpu(tensor_ptr, nRows, nCols, q_tensor_ptr);
//     return q_tensor;
// }

torch::Tensor row_wise_quantization_gpu(torch::Tensor& d_tensor){
    if(d_tensor.dtype() != torch::kFloat16){
        std::cerr << "Tensor must have float16 dtype!" << std::endl;
        return torch::zeros(1);
    }
    if(!d_tensor.device().is_cuda()){
        std::cerr << "Tensor must be on device!" << std::endl;
        return torch::zeros(1);
    }

    int nRows = d_tensor.size(0);
    int nCols = d_tensor.size(1);

    torch::Tensor d_rowsScale = torch::empty(nRows, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
    torch::Tensor d_q_tensor = torch::empty({nRows, nCols}, torch::TensorOptions().dtype(torch::kInt8).device(torch::kCUDA));

    __half *d_tensor_ptr = reinterpret_cast<__half*>(d_tensor.data_ptr<at::Half>());
    __half *d_rowsScale_ptr = reinterpret_cast<__half*>(d_rowsScale.data_ptr<at::Half>());
    int8_t *d_q_tensor_ptr = d_q_tensor.data_ptr<int8_t>();

    rowWiseQuant8bits_gpu2(d_tensor_ptr, nRows, nCols, d_q_tensor_ptr, d_rowsScale_ptr);
    return d_q_tensor;
}

torch::Tensor column_wise_quantization_gpu(torch::Tensor& tensor){
    auto q_tensor = torch::zeros_like(tensor, torch::kInt8);

    int nRows = tensor.size(0);
    int nCols = tensor.size(1);

    float *tensor_ptr = tensor.data_ptr<float>();
    int8_t *q_tensor_ptr = q_tensor.data_ptr<int8_t>();

    columnWiseQuant8bits_gpu(tensor_ptr, nRows, nCols, q_tensor_ptr);
    return q_tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("symmetric_quantization_gpu", &symmetric_quantization_gpu, 
          "Symmetric quantization on GPU", py::arg("array"));
    m.def("row_wise_quantization_gpu", &row_wise_quantization_gpu,
        "Row wise quantization on GPU");
    m.def("column_wise_quantization_gpu", &column_wise_quantization_gpu,
        "Column wise quantization on GPU");
}