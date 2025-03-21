#pragma once
#include "COOMatrixGPU.cuh"

void spmvCoo_gpu(COOMatrixGPU& mat, size_t nRows, size_t nCols, float *x, float *y);
void spmvCSR_gpu(CSRMatrixGPU& mat, size_t nRows, size_t nCols, float *x, float *y);