#include <iostream>
#include <random>
#include <chrono>

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


float** createMatrix(int nRows, int nCols){
    float **matrix = new float*[nRows];
    for(int i = 0; i < nRows; ++i)
        matrix[i] = new float[nCols];
    return matrix;
}

void initMatrix(float **matrix, int nRows, int nCols){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist;
    for(int i = 0; i < nRows; ++i){
        for(int j = 0; j < nCols; ++j){
            matrix[i][j] = dist(gen);
        }
    }
}

void printMatrix(float **matrix, int nRows, int nCols){
    for(int i = 0; i < nRows; ++i){
        for(int j = 0; j < nCols; ++j){
            std::cout << matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

float* flattenMatrix(float **mat, int nRows, int nCols){
    float *flatMat = new float[nRows * nCols];
    for(int i = 0; i < nRows; ++i){
        for(int j = 0; j < nCols; ++j){
            flatMat[i * nCols + j] = mat[i][j];
        }
    }
    return flatMat;
}

void freeMatrix(float **mat, int nRows){
    for(int i = 0; i < nRows; ++i)
        delete[] mat[i];
    delete mat;
}

void mm_cpu(float **mat1, float **mat2, float **result, int nRows1, int nCols1, int nRows2, int nCols2){
    if(nCols1 != nRows2){
        std::cerr << "ERROR: nCols1 != nRows1. These must be equal to be able to apply matrix multiplcation!" << std::endl;
        exit(EXIT_FAILURE);
    }
    for(int i = 0; i < nRows1; ++i){
        for(int j = 0; j < nCols2; ++j){
            result[i][j] = 0.f;
            for(int k = 0; k < nCols1; ++k){
                result[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }
}

__global__
void mmNaive_gpu(float *mat1, float *mat2, float *result, int nRows1, int nCols1, int nRows2, int nCols2){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;

    if(i < nRows1 && j < nCols2){
        float sum = 0.f;
        for(int k = 0; k < nCols1; ++k)
            sum += mat1[i * nCols1 + k] * mat2[k * nCols2 + j];
        result[i * nCols2 + j] = sum;
    }
}

int main(){
    int nRows1 = 512, nRows2 = 256, nCols1 = 256, nCols2 = 512;
    float **mat1 = createMatrix(nRows1, nCols1);
    float **mat2 = createMatrix(nRows2, nCols2);
    float **result = createMatrix(nRows1, nCols2);
    
    initMatrix(mat1, nRows1, nCols1);
    initMatrix(mat2, nRows2, nCols2);

    // Device vars
    float *d_mat1, *d_mat2, *d_result;
    size_t size1 = nRows1 * nCols1 * sizeof(float);
    size_t size2 = nRows2 * nCols2 * sizeof(float);
    size_t sizer = nRows1 * nCols2 * sizeof(float);

    CUDA_CHECK_ERROR(cudaMalloc(&d_mat1, size1));
    CUDA_CHECK_ERROR(cudaMalloc(&d_mat2, size2));
    CUDA_CHECK_ERROR(cudaMalloc(&d_result, sizer));

    // Flatten host 2d arrays to copy them to device memory
    float *mat1Flat = flattenMatrix(mat1, nRows1, nCols1);
    float *mat2Flat = flattenMatrix(mat2, nRows2, nCols2);
    float *resultGPU = flattenMatrix(result, nRows1, nCols2);

    // Copy matrices to device
    CUDA_CHECK_ERROR(cudaMemcpy(d_mat1, mat1Flat, size1, cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_mat2, mat2Flat, size2, cudaMemcpyHostToDevice));

    // Create events for computing runtime
    cudaEvent_t start, end;
    CUDA_CHECK_ERROR(cudaEventCreate(&start));
    CUDA_CHECK_ERROR(cudaEventCreate(&end));

    // Launch Kernel
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((nRows1 + threadsPerBlock.x - 1) / threadsPerBlock.x, (nCols2 + threadsPerBlock.y - 1) / threadsPerBlock.y);
    CUDA_CHECK_ERROR(cudaEventRecord(start));

    mmNaive_gpu<<<numBlocks, threadsPerBlock>>>(d_mat1, d_mat2, d_result, nRows1, nCols1, nRows2, nCols2);

    CUDA_CHECK_ERROR(cudaEventRecord(end));
    CUDA_CHECK_ERROR(cudaEventSynchronize(end));
    CUDA_KERNEL_CHECK_ERROR();

    // Copy result to host
    CUDA_CHECK_ERROR(cudaMemcpy(resultGPU, d_result, sizer, cudaMemcpyDeviceToHost));

    // Kernel runtime
    float gpuTimeMilliSeconds = 0.f;
    CUDA_CHECK_ERROR(cudaEventElapsedTime(&gpuTimeMilliSeconds, start, end));

    // CPU version for testing
    auto cpuStart = std::chrono::high_resolution_clock::now();
    mm_cpu(mat1, mat2, result, nRows1, nCols1, nRows2, nCols2);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    auto cpuTimeMicroseconds = std::chrono::duration_cast<std::chrono::microseconds>(cpuEnd - cpuStart).count();

    // Print timing
    std::cout << "CPU function runtime: " << cpuTimeMicroseconds << " us" << std::endl;
    std::cout << "GPU kernel runtime: " << gpuTimeMilliSeconds * 1000 << " us" << std::endl;
    std::cout << "Speedup: " << cpuTimeMicroseconds / (gpuTimeMilliSeconds * 1000) << std::endl;

    // Check output matching
    std::cout << "Comparing outputs..." << std::endl;
    bool testFailed = false;
    for(int i = 0; i < nRows1; ++i){
        for(int j = 0; j < nCols2; ++j){
            if((result[i][j] - resultGPU[i * nCols2 + j]) > 1e-4){
                std::cout << "Mismatch at index " << i << "," << j << ": Expected " << result[i][j] << ", Found " << resultGPU[i * nCols2 + j] << std::endl;
                testFailed = true;
            }
        }
    }
    if(testFailed)
        std::cout << "\033[31mTEST FAILED\033[0m" << std::endl;
    else
        std::cout << "\033[32mTEST PASSED!\033[0m" << std::endl;

    // Free memory
    cudaFree(d_mat1);
    cudaFree(d_mat2);
    cudaFree(d_result);
    freeMatrix(mat1, nRows1);
    freeMatrix(mat2, nRows2);
    freeMatrix(result, nRows1);
    free(mat1Flat);
    free(mat2Flat);
    free(resultGPU);
    return 0;
}