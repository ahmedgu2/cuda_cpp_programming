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
 * Luong model is as follows:
 * - encoder: unidirectional (simple).
 * - attention score: biliniaire function, score(h_t, s_k) = h^T_{t} x W x s_k
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
void bahdanauAttScore_gpu(){
    
}