#include <iostream>
#include <random>
#include <cmath>
#include <vector>
#include <fstream>
#include <string>

// Utils
void initMatrix(float *matrix, int nRows, int nCols, unsigned int seed = 42){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 255);
    for(int i = 0; i < nRows; ++i){
        for(int j = 0; j < nCols; ++j){
            matrix[i * nCols + j] = dist(gen);
        }
    }
}

int clamp(const int val, const int min, const int max){
    if(val < min) return min;
    if(val > max) return max;
    return val;
}

void printMatrix(float *matrix, int nRows, int nCols){
    for(int i = 0; i < nRows; ++i){
        for(int j = 0; j < nCols; ++j){
            std::cout << matrix[i * nRows + j] << " ";
        }
        std::cout << std::endl;
    }
}

void saveMatrixCSV(const float *matrix, const int nRows, const int nCols, const std::string &filename) {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Error opening file for writing!" << std::endl;
        return;
    }

    for (int i = 0; i < nRows; ++i) {
        for (int j = 0; j < nCols; j++) {
            file << matrix[i * nRows + j];
            if (j < nCols - 1) 
                file << ",";  // Separate columns with commas
        }
        file << "\n";
    }

    file.close();
}

// Gaussian Blur implementation
float normal_pdf2D(const float x, const float y, const float std){
    const float pi = 3.141592653589;
    const float coeff = 1 / (2 * pi * std * std);
    const float exponent = - 0.5 * ((x * x + y * y) / (std * std));
    return coeff * exp(exponent);
}

void initKernel2D(std::vector<float>& kernel, const int kernelSize){
    // Since kernelSize is odd, it's written as 2k + 1. k in here is the center of the kernel grid.
    // To apply the gaussian correctly, we need to express (i, j) relative to the center of the grid (k, k).
    const int k = (kernelSize - 1) / 2;  
    float sum = 0.f; // Normalization 
    for(int i = 0; i < kernelSize; ++i){
        for(int j = 0; j < kernelSize; ++j){
            kernel[i * kernelSize + j] = normal_pdf2D(i - k, j - k, 1.0);
            sum += kernel[i * kernelSize + j];
        }
    }
    for(int i = 0; i < kernelSize; ++i){
        for(int j = 0; j < kernelSize; ++j){
            kernel[i * kernelSize + j] /= sum;
        }
    }
}

float* convolute(float *mat, const int nRows, const int nCols, float *kernel, const int kernelSize){
    /**
     * Apply convolution (cross-correclation to be exact as the kernel is not flipped).
     * Padding strategy : Edge extension, i.e. handle out of boundary cells take on the nearest cell value.
     */
    float *bluredMat = new float[nRows * nCols];
    const int k = (kernelSize - 1) / 2;
    for(int i = 0; i < nRows; ++i){
        for(int j = 0; j < nCols; ++j){
            float sum = 0.f;

            for(int ki = -k; ki <= k; ++ki){
                for(int kj = -k; kj <= k; ++kj){
                    int curi = clamp(i + ki, 0, nRows - 1);
                    int curj = clamp(j + kj, 0, nCols - 1);
                    sum += mat[curi * nCols + curj] * kernel[(ki + k) * kernelSize + (kj + k)];
                }
            }
            bluredMat[i * nCols + j] = sum;
        }
    }
    return bluredMat;
}

float* gaussianBlur_cpu(
    float *mat,
    const int nRows,
    const int nCols,
    const int kernelSize
){
    if(kernelSize % 2 == 0){
        std::cerr << "Kernel size should be odd!" << std::endl;
        exit(EXIT_FAILURE);
    }
    std::vector<float> kernel(kernelSize * kernelSize);
    
    initKernel2D(kernel, kernelSize);
    // printMatrix(kernel.data(), kernelSize, kernelSize);
    return convolute(mat, nRows, nCols, kernel.data(), kernelSize);
}


int main(){
    const int nRows = 16, nCols = 16;
    const int kernelSize = 3;
    float *mat = new float[nRows * nCols];

    initMatrix(mat, nRows, nCols);
    //printMatrix(mat, nRows, nCols);

    float *bluredMat = gaussianBlur_cpu(mat, nRows, nCols, kernelSize);

    // Save matrices to do some testing with cv2 in python
    saveMatrixCSV(bluredMat, nRows, nCols, "data/bluredMatrix.csv");
    saveMatrixCSV(mat, nRows, nCols, "data/matrix.csv");

    delete [] bluredMat;

    return 0;
}