#pragma once
#include <cuda_fp16.h>

void rowWiseQuant8bits_gpu(__half *X, size_t nRows, size_t nCols, int8_t *q_X);
void columnWiseQuant8bits_gpu(float *X, size_t nRows, size_t nCols, int8_t *q_X);

template <typename IN_TYPE, typename OUT_TYPE>
void matmul_gpu(IN_TYPE *X1, size_t nRows1, size_t nCols1, IN_TYPE *X2, size_t nRows2, size_t nCols2, OUT_TYPE *result);
void outliersColumns_gpu(float *X, size_t nRows, size_t nCols, bool *isOutlierCol, float threshold = 6);