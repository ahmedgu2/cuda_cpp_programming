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

void rowWiseQuant8bits_cpu(float *X, size_t nRows, size_t nCols, int8_t *q_X){
    for(int row = 0; row < nRows; ++row){
        int start_row = row * nCols;
        int end_row = row * nCols + nCols;
        float abs_max = *std::max_element(
            X + start_row,
            X + end_row,
            [](float a, float b){
                return abs(a) < abs(b);
            }
        );
        abs_max = abs(abs_max);
        float S = 127 / abs_max;
        for(int col = 0; col < nCols; ++col){
            q_X[row * nCols + col] = roundf(X[row * nCols + col] * S);
        }
    }
}

void columnWiseQuant8bits_cpu(float *X, size_t nRows, size_t nCols, int8_t *q_X){
    for(int col = 0; col < nCols; ++col){
        float abs_max = 0;
        for(int row = 0; row < nRows; ++row){
            abs_max = fmax(abs_max, abs(X[row * nCols + col]));
        }
        float S = 127 / abs_max;
        for(int row = 0; row < nRows; ++row){
            q_X[row * nCols + col] = roundf(X[row * nCols + col] * S);
        }
    }
}

void outliersColumns_cpu(float *X, size_t nRows, size_t nCols, bool *isOutlierCol, float threshold){
    for(int row = 0; row < nRows; ++row){
        for(int col = 0; col < nCols; ++col){
            if(X[row * nCols + col] >= threshold){
                isOutlierCol[col] = 1;
            }
        }
    }
}