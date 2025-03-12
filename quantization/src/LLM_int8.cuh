#pragma once

void rowWiseQuant8bits_gpu(float *X, size_t nRows, size_t nCols, int8_t *q_X);
void columnWiseQuant8bits_gpu(float *X, size_t nRows, size_t nCols, int8_t *q_X);
void dequentize(int32_t *q_X, size_t nRows, size_t nCols, float *rowsScale, float *columnsScale, float *X);