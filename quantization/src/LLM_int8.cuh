#pragma once

void rowWiseQuant8bits_gpu(float *X, size_t nRows, size_t nCols, int8_t *q_X);