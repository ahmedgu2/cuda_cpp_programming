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

float* bilinearAttScore_gpu(float *h_t, float *W, float *S, uint32_t d, uint32_t m){
    /**
     * This function implements the bilinear attention score of Luong's model.
     * score(h_t, s_k) = h^T_{t} x W x s_k
     * We can compute all scores (s_k for k in 1..m) in one go as folows:

     *      scores(h_t, S) = h^T_{t} x W x S

     *  with - `S` a matrix of shape (d, m), where each colum repesent s_k, for k in 1..m
     *       - `W` a matrix of shape (d, d)
     *       - `h^T_{t}` a vector of shape (1, d)
     * 
     * Return:
     *  - scores: Attention scores for s(h_t, s_k), for k in 1..m
     */

    float *d_h_t, *d_W, *d_S, *d_h_W, *d_scores;
    
    // Allocate on device
    CUDA_CHECK_ERROR(cudaMalloc(&d_h_t, d * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_W, d * d * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_S, d * m * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_h_W, d * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_scores, m * sizeof(float)));

    // Copy from host to device
    CUDA_CHECK_ERROR(cudaMemcpy(d_h_t, h_t, d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_W, W, d * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_S, S, d * m * sizeof(float), cudaMemcpyHostToDevice));

    // Run Kernel
    const dim3 blockDim(TILE_DIM, 1);
    dim3 gridSize((TILE_DIM + d - 1) / TILE_DIM, 1);
    // h_W =  h^T_{t} x W
    matmul<<<gridSize, blockDim>>>(d_h_t, 1, d, d_W, d, d, d_h_W);
    CUDA_KERNEL_CHECK_ERROR();
    // result = h_W x S
    gridSize = {(TILE_DIM + m - 1) / TILE_DIM, 1}; // Change dimension
    matmul<<<gridSize, blockDim>>>(d_h_W, 1, d, d_S, d, m, d_scores);
    CUDA_KERNEL_CHECK_ERROR();

    // Copy to cpu
    float *scores = new float[m];
    CUDA_CHECK_ERROR(cudaMemcpy(scores, d_scores, m * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    cudaFree(d_h_t);
    cudaFree(d_S);
    cudaFree(d_W);
    cudaFree(d_h_W);

    return scores;
}

int main(){

    const uint32_t d = 256, m = 32;
    float S[d * m], W[d * d], h_t[d];
    
    initVector(S, d * m);
    initVector(W, d * d);
    initVector(h_t, d);

    float* scores_gpu = bilinearAttScore_gpu(h_t, W, S, d, m);

    // std::cout << std::fixed << std::setprecision(7) << score_cpu << " " << scores_gpu << std::endl;
    // if(std::abs(scores_cpu - scores_gpu) > 1e-3){
    //     std::cerr << "\033[31mTEST FAILED\033[0m" << std::endl;
    //     exit(EXIT_FAILURE);
    // }
    // std::cout << "\033[32mTEST PASSED!\033[0m" << std::endl;

    delete[] scores_gpu;
    return 0;
}