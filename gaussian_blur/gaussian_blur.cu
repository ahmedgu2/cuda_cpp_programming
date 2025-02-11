#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <random>


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
__device__ 
float normal_pdf2D_gpu(const float x, const float y, const float mean, const float std){
    const float pi = 3.141592653589;
    const float coeff = 1 / (2 * pi * std * std);
    const float exponent = - 0.5 * ((x * x + y * y) / (std * std));
    return coeff * expf(exponent);
}

__device__
int clamp(const int x, const int min, const int max){
   if(x < min) return min;
   if(x > max) return max;
   return x;
}

__global__
void initKernel2D_gpu(float *kernel, const int kernelSize, float *sum){
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if(col < kernelSize && row < kernelSize){
        int indx = row * kernelSize + col;
        const int k = (kernelSize - 1) / 2;

        kernel[indx] = normal_pdf2D_gpu(row - k, col - k, 0.f, 1.f);

        // Accumulate the sum using atomics
        atomicAdd(sum, kernel[indx]);
    }
}

__global__
void normalizeKernel2D_gpu(float *kernel, const int kernelSize, float *sum){
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if(col < kernelSize && row < kernelSize){
        int indx = row * kernelSize + col;
        kernel[indx] /= *sum;
    }
}

__global__ 
void convoluteNaive_gpu(float *result, float *mat, const int nRows, const int nCols, float *kernel, const int kernelSize){
    // We're using edge expension padding, so dimension of result and mat are equal.
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if(row < nRows && col < nCols){
        int k = (kernelSize - 1) / 2;
        int indx = row * nCols + col;
        
        result[indx] = 0;

        for(int i = -k; i <= k; ++i){
            for(int j = -k; j <= k; ++j){
                int curRow = clamp(row + i, 0, nRows - 1);
                int curCol = clamp(col + j, 0, nCols - 1);
                result[indx] += mat[curRow * nCols + curCol] * kernel[(i + k) * kernelSize + (j + k)];
            }
        }
    }
}


void gaussianBlur_gpu(
    float *mat,
    float *result,
    const int nRows,
    const int nCols,
    const int kernelSize
){
   // We consider that the KernelSize in here is at most 7x7, thus it fits in a single block.
    float *d_mat, *d_result;
    float *d_kernel;
    float *d_sum;

    CUDA_CHECK_ERROR(cudaMalloc(&d_kernel, kernelSize * kernelSize * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_result, nRows * nCols * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_mat, nRows * nCols * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_sum, sizeof(float)));

    // Init sum
    CUDA_CHECK_ERROR(cudaMemset(d_sum, 0, sizeof(float)));
    CUDA_CHECK_ERROR(cudaMemcpy(d_mat, mat, nRows * nCols * sizeof(float), cudaMemcpyHostToDevice));

    // Init the kernel
    dim3 kernelThreads(kernelSize, kernelSize);
    initKernel2D_gpu<<<1, kernelThreads>>>(d_kernel, kernelSize, d_sum);
    cudaDeviceSynchronize();
    CUDA_KERNEL_CHECK_ERROR();

    normalizeKernel2D_gpu<<<1, kernelThreads>>>(d_kernel, kernelSize, d_sum);
    cudaDeviceSynchronize();
    CUDA_KERNEL_CHECK_ERROR();

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((nCols + threadsPerBlock.x - 1) / threadsPerBlock.x, (nRows + threadsPerBlock.y - 1) / threadsPerBlock.y);
    convoluteNaive_gpu<<<numBlocks, threadsPerBlock>>>(d_result, d_mat, nRows, nCols, d_kernel, kernelSize);
    CUDA_KERNEL_CHECK_ERROR();

    CUDA_CHECK_ERROR(cudaMemcpy(result, d_result, nRows * nCols * sizeof(float), cudaMemcpyDeviceToHost));

    // float *kernel = new float[kernelSize * kernelSize];
    // CUDA_CHECK_ERROR(cudaMemcpy(kernel, d_kernel, kernelSize * kernelSize * sizeof(float), cudaMemcpyDeviceToHost));
    // printMatrix(kernel, kernelSize, kernelSize);
    // std::cout << " ######################### \n";
    // printMatrix(mat, nRows, nCols);
    // std::cout << " ######################### \n";
    // printMatrix(result, nRows, nCols);
    // delete[] kernel;

    cudaFree(d_mat);
    cudaFree(d_kernel);
    cudaFree(d_result);
    cudaFree(d_sum);
}

int main(){

    int nRows = 8, nCols = 8;
    int kernelSize = 3;
    float *mat = new float[nRows * nCols];
    float *result = new float[nRows * nCols];

    initMatrix(mat, nRows, nCols);

    gaussianBlur_gpu(mat, result, nRows, nCols, kernelSize);

    delete[] mat;
    delete[] result;
}