#include "cuda_macros.cuh"
#include "COOMatrixGPU.cuh"

__global__
void spmv_coo(COOMatrixGPU mat, float *x, float *y){
    int indx = threadIdx.x + blockDim.x * blockIdx.x;
    if(indx < mat.getNumNonZeros()){
        uint32_t row = mat.getRowIndx(indx);
        uint32_t col = mat.getColIndx(indx);
        float val = mat.getValue(indx);
        atomicAdd(&y[row], val * x[col]);
    }
}

void spmvCoo_gpu(COOMatrixGPU& mat, size_t nRows, size_t nCols, float *x, float *y){
    float *d_x, *d_y;

    CUDA_CHECK_ERROR(cudaMalloc(&d_x, nCols * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_y, nRows * sizeof(float)));

    CUDA_CHECK_ERROR(cudaMemset(d_y, 0, nRows * sizeof(float)));

    CUDA_CHECK_ERROR(cudaMemcpy(d_x, x, nCols * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_y, y, nRows * sizeof(float), cudaMemcpyHostToDevice));

    size_t length = mat.getNumNonZeros();
    int BLOCK_DIM = 1024;
    int GRID_DIM = (length + BLOCK_DIM - 1) / BLOCK_DIM;
    // FIXME: Passing by value calls the desctructor after kernel finishes, which will make mat unusable afterwards.
    spmv_coo<<<GRID_DIM, BLOCK_DIM>>>(mat, d_x, d_y);
    CUDA_KERNEL_CHECK_ERROR();
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR(cudaMemcpy(y, d_y, nRows * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK_ERROR(cudaFree(d_x));
    CUDA_CHECK_ERROR(cudaFree(d_y));
}