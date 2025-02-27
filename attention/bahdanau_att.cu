#include <iostream>
#include <random>
#include <iomanip>

#define TILE_DIM 32

/******************* Utility macros and functions ********************** */
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
void initVector(float *vector, const int length, unsigned int seed = 42){
    std::mt19937 gen(seed);
    // std::uniform_int_distribution<int> dist(1, 255);
    std::normal_distribution<float> dist;
    for(int i = 0; i < length; ++i){
        vector[i] = (float)dist(gen);
    }
}

void printVector(float *vector, const int length){
    for(int i = 0; i < length; ++i){
        std::cout << vector[i] << " ";
    }
    std::cout << std::endl;
}

/**************** Luong attention implementation ************* */
/**
 * Badhanau model is as follows:
 * - encoder: bi-directional.
 * - attention score: biliniaire function, score(h_t, s_k) = W^T_2 x tanh(W_1 [h_{t-1}, s_k])
 * - attention applied between RNN decoder at step t and prediction for this step.
 */
__global__
void matmul(float *mat1, uint32_t nRows1, uint32_t nCols1, float *mat2, uint32_t nRows2, uint32_t nCols2, float *output){
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y; 
    int tx = threadIdx.x, ty = threadIdx.y;

    __shared__ float mat1Tile[TILE_DIM][TILE_DIM];
    __shared__ float mat2Tile[TILE_DIM][TILE_DIM];

    float outVal = 0.f;
    
    // Load tiles in shared memory
    for(int tile = 0; tile < ((nCols1 + TILE_DIM - 1) / TILE_DIM); ++tile){

        if(tile * TILE_DIM + tx < nCols1 && row < nRows1)
            mat1Tile[ty][tx] = mat1[row * nCols1 + (tile * TILE_DIM + tx)];
        else
            mat1Tile[ty][tx] = 0.f;
        if(tile * TILE_DIM + ty < nRows2 && col < nCols2)
            mat2Tile[ty][tx] = mat2[(TILE_DIM * tile + ty) * nCols2 + col];
        else
            mat2Tile[ty][tx] = 0.f;
        __syncthreads();

        // Compute the partial dot product for that tile
        for(int k = 0; k < TILE_DIM; ++k){
            outVal += mat1Tile[ty][k] * mat2Tile[k][tx];
        }
        __syncthreads();
    }
    if(row < nRows1 && col < nCols2)
        output[row * nCols2 + col] = outVal;
}

__global__
void concat(float *x, const int len_x, float *y, const int len_y, float *output, float len_out){
    for(int indx = threadIdx.x + blockDim.x * blockIdx.x; indx < len_out; indx += gridDim.x * blockDim.x){
        if(indx < len_x)
            output[indx] = x[indx];
        if(indx >= len_x)
            output[indx] = y[indx - len_x];
    }
}

__global__
void tanh_(float *mat, const int nRows, const int nCols, float *output){
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;

    if(row < nRows && col < nCols){
        output[row * nCols + col] = tanhf(mat[row * nCols + col]);
    }
}


void bahdanauAttScore_gpu(
    float *w_2_T,
    float *W_1,
    float *h,
    float *S,
    const int d,
    const int m
){
    /**
     * 
     */

    float *d_w_2_T, *d_W_1, *d_h, *d_S, *d_concat, *d_matmul;

        // Allocate memory on GPU
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_w_2_T, d * sizeof(float)));       // Shape: (1, d)
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_W_1, d * 2 * d * sizeof(float)));    // Shape: (d, 2d)
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_h, d * sizeof(float)));         // Shape: (d, 1)
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_S, d * sizeof(float)));     // Shape: (d, 1)
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_matmul, d * sizeof(float)));     // Shape: (d, 1)
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_concat, 2 * d * sizeof(float)));     // Shape: (2d, 1)

    // Copy memory from host to device
    CUDA_CHECK_ERROR(cudaMemcpy(d_w_2_T, w_2_T, d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_W_1, W_1, d * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_h, h, d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_S, S, d * m * sizeof(float), cudaMemcpyHostToDevice));

    // TODO: Implement kernel execution
    int BLOCK_DIM = 256;
    dim3 gridSize((d + BLOCK_DIM - 1) / BLOCK_DIM);
    concat<<<gridSize, BLOCK_DIM>>>(d_h, d, d_S, d, d_concat, 2 * d);
    CUDA_KERNEL_CHECK_ERROR();

    dim3 numThreads(BLOCK_DIM, BLOCK_DIM);
    matmul<<<gridSize, numThreads>>>(d_W_1, 2 * d, d, d_concat, 2 * d, 1, d_matmul);
    CUDA_KERNEL_CHECK_ERROR();
    // dim3 numThreads(BLOCK_DIM, BLOCK_DIM);

    tanh_<<<gridSize, numThreads>>>(d_matmul, d, 1, d_matmul);
    CUDA_KERNEL_CHECK_ERROR();
    
    // Free allocated memory (should be done after kernel execution)
    CUDA_CHECK_ERROR(cudaFree(d_w_2_T));
    CUDA_CHECK_ERROR(cudaFree(d_W_1));
    CUDA_CHECK_ERROR(cudaFree(d_h));
    CUDA_CHECK_ERROR(cudaFree(d_S));
    CUDA_CHECK_ERROR(cudaFree(d_concat));
    CUDA_CHECK_ERROR(cudaFree(d_matmul));
}

/******************** Unit testing ******************* */
void concat_cpu(float *x, const int len_x, float *y, const int len_y, float *output, const int len_out){
    for(int i = 0; i < len_out; i++){
        if(i < len_x)
            output[i] = x[i];
        else
            output[i] = y[i - len_x];
    }
}

void test_concat(){
    const int len_x = 1e5, len_y = 1024;
    const int len_out = len_x + len_y;
    float x[len_x], y[len_y];
    float *output_cpu = new float[len_out];
    float *output_gpu = new float[len_out];

    initVector(x, len_x);
    initVector(y, len_y);

    float *d_x, *d_y, *d_output;

    CUDA_CHECK_ERROR(cudaMalloc(&d_x, len_x * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_y, len_y * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_output, len_out * sizeof(float)));
    
    CUDA_CHECK_ERROR(cudaMemcpy(d_x, x, len_x * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_y, y, len_y * sizeof(float), cudaMemcpyHostToDevice));

    int BLOCK_DIM = 256;
    int gridSize = (len_out + BLOCK_DIM - 1) / BLOCK_DIM;
    concat<<<BLOCK_DIM, gridSize>>>(d_x, len_x, d_y, len_y, d_output, len_out);
    CUDA_KERNEL_CHECK_ERROR();

    CUDA_CHECK_ERROR(cudaMemcpy(output_gpu, d_output, len_out * sizeof(float), cudaMemcpyDeviceToHost));

    concat_cpu(x, len_x, y, len_y, output_cpu, len_out);

    for(int i = 0; i < len_out; ++i){
        if(output_cpu[i] != output_gpu[i]){
            std::cerr << "Mismatch at index " << i << ". Expected " << output_cpu[i] << ", Found " << output_gpu[i] << std::endl;
            std::cerr << "\033[31mTEST FAILED\033[0m" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    std::cout << "\033[32mTEST PASSED!\033[0m" << std::endl;

    delete[] output_cpu;
    delete[] output_gpu;
}

int main(){
    test_concat();
}