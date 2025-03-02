#pragma once
#include <vector>

void initVector(float *vector, const int length, unsigned int seed = 42);
void printVector(float *vector, const int length);
void printMatrix(std::vector<float>& mat, const int nRows, const int nCols);