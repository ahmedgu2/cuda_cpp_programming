
#include <iostream>
#include <random>
#include <iomanip>


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
void matmul(float *mat1, int nRows1, int nCols1, float *mat2, int nRows2, int nCols2, float *output){
    int indx_x = threadIdx.x + blockDim.x * blockIdx.x;
    int indx_y = threadIdx.y + blockDim.y * blockIdx.y; 
}

void bilinearAttScore(){
    /**
     * This function implements the bilinear attention score of Luong's model.
     * score(h_t, s_k) = h^T_{t} x W x s_k
     * We can compute all scores (s_k for k in 1..m) in one go as folows:

     *      scores(h_t, S) = h^T_{t} x W x S

     *  with - `S` a matrix of shape (d, m)
     *       - `W` a matrix of shape (d, d)
     *       - `h^T_{t}` a vector of shape (1, d)
     */

}
