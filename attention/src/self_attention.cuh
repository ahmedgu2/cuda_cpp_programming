#pragma once

float* softmax2D_gpu(float *X, const int nRows, const int nCols);
float *matmul_gpu(
    float *mat1,
    const int nRows1,
    const int nCols1,
    float *mat2,
    const int nRows2,
    const int nCols2
);

float* selfAttention_gpu(
    float *Q,
    float *K,
    float *V,
    float *W_q,
    float *W_k,
    float *W_v,
    const int seq_len,
    const int dim_emb
);