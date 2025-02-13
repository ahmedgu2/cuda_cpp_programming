#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <random>

#define KERNEL_SIZE 7
#define TILE_DIM 32
__constant__ float kernel[KERNEL_SIZE * KERNEL_SIZE];


#define CUDA_CHECK_ERROR(callResult) do{ \
    cudaError_t error = callResult; \
    if(error != cudaSuccess){ \
        std::cout << "Error in " << __FILE__ << " at line " << __LINE__ << "\n" << cudaGetErrorString(error); \
        exit(EXIT_FAILURE); \
    } \
}while(0)

#define CUDA_KERNEL_CHECK_ERROR() do{ \
    cudaError_t error = cudaGetLastError(); \
    if(error != cudaSuccess){ \
        std::cerr << "----CUDA ERROR in " << __FILE__ << " at line " << __LINE__ << "\n" << cudaGetErrorString(error) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}while(0)


// Utils
void initMatrix(float *matrix, int nRows, int nCols, unsigned int seed = 42){
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(0, 255);
    for(int i = 0; i < nRows; ++i){
        for(int j = 0; j < nCols; ++j){
            matrix[i * nCols + j] = dist(gen);
        }
    }
}

void printMatrix(float *matrix, int nRows, int nCols){
    for(int i = 0; i < nRows; ++i){
        for(int j = 0; j < nCols; ++j){
            std::cout << matrix[i * nRows + j] << " ";
        }
        std::cout << std::endl;
    }
}

// GPU version
float normal_pdf2D(float x, const float y, const float std){
    const float pi = 3.141592653589;
    const float coeff = 1 / (2 * pi * std * std);
    const float exponent = - 0.5 * ((x * x + y * y) / (std * std));
    return coeff * exp(exponent);
}

void initKernel2D(float *kernel, const int kernelSize){
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

__device__
int clamp(const int x, const int min, const int max){
   if(x < min) return min;
   if(x > max) return max;
   return x;
}

__global__ 
void convoluteOptimized_gpu(float *result, float *mat, const int nRows, const int nCols){
    // We're using edge expension padding, so dimension of result and mat are equal.
    __shared__ float tile[TILE_DIM + KERNEL_SIZE - 1][TILE_DIM + KERNEL_SIZE - 1];
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int k = (KERNEL_SIZE - 1) / 2;
    // Shared memory indices
    int shared_x = tx + k;
    int shared_y = ty + k;

    // Load tile in shared memory
    if(row < nRows && col < nCols){
        tile[shared_y][shared_x] = mat[row * nCols + col];
    }else{
        int clampedCol = clamp(col, 0, nCols - 1);
        int clampedRow = clamp(row, 0, nRows);
        tile[ty][tx] = mat[clampedRow * nCols + clampedCol];
    }
    __syncthreads();


    if(row < nRows && col < nCols){
        int indx = row * nCols + col;
        result[indx] = 0;

        for(int i = -k; i <= k; ++i){
            for(int j = -k; j <= k; ++j){
                int curRow = clamp(row + i, 0, nRows - 1);
                int curCol = clamp(col + j, 0, nCols - 1);
                result[indx] += mat[curRow * nCols + curCol] * kernel[(i + k) * KERNEL_SIZE + (j + k)];
            }
        }
    }
}

void gaussianBlurOptimized_gpu(
    float *mat,
    float *result,
    const int nRows,
    const int nCols
){
   // We consider that the KernelSize in here is at most 7x7, thus it fits in a single block.
    float *d_mat, *d_result;
    float *d_sum;
    float h_kernel[KERNEL_SIZE * KERNEL_SIZE];

    initKernel2D(h_kernel, KERNEL_SIZE);

    CUDA_CHECK_ERROR(cudaMalloc(&d_result, nRows * nCols * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_mat, nRows * nCols * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_sum, sizeof(float)));

    // Init sum
    CUDA_CHECK_ERROR(cudaMemset(d_sum, 0, sizeof(float)));
    CUDA_CHECK_ERROR(cudaMemcpy(d_mat, mat, nRows * nCols * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpyToSymbol(kernel, h_kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float)));

    dim3 threadsPerBlock(TILE_DIM, TILE_DIM);
    dim3 numBlocks((nCols + threadsPerBlock.x - 1) / threadsPerBlock.x, (nRows + threadsPerBlock.y - 1) / threadsPerBlock.y);
    convoluteOptimized_gpu<<<numBlocks, threadsPerBlock>>>(d_result, d_mat, nRows, nCols);
    CUDA_KERNEL_CHECK_ERROR();

    CUDA_CHECK_ERROR(cudaMemcpy(result, d_result, nRows * nCols * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_mat);
    cudaFree(d_result);
    cudaFree(d_sum);
}

int main(){

    int nRows = 1024, nCols = 1024;
    float *mat = new float[nRows * nCols];
    float *result = new float[nRows * nCols];

    initMatrix(mat, nRows, nCols);

    gaussianBlurOptimized_gpu(mat, result, nRows, nCols);

    delete[] mat;
    delete[] result;
}