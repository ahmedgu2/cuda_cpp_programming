#include "self_attention.cuh"
#include "AttentionLayer.h"
#include "cuda_macros.cuh"
#include "cuda_utils.cuh"
#include "utils.h"
#include <random>
#include <vector>

AttentionLayer::AttentionLayer(uint32_t seq_len, uint32_t dim_emb, std::string device){
    this->device = device;
    this->dim_emb = dim_emb;
    this->seq_len = seq_len;
    initWeights();
}

AttentionLayer::~AttentionLayer(){
    if(device == "cuda"){
        freeCudaMemory(W_q);
        freeCudaMemory(W_k);
        freeCudaMemory(W_v);
    }
}

void AttentionLayer::initWeights(){
    size_t weightSize = dim_emb * dim_emb;
    std::vector<float> W_q_tmp(weightSize);
    std::vector<float> W_k_tmp(weightSize);
    std::vector<float> W_v_tmp(weightSize);

    // init
    initArrayXavier(W_q_tmp.data(), weightSize, dim_emb, dim_emb);
    initArrayXavier(W_k_tmp.data(), weightSize, dim_emb, dim_emb);
    initArrayXavier(W_v_tmp.data(), weightSize, dim_emb, dim_emb);

    // Allocate weights
    if(device == "cuda"){
        allocateCudaMemory(&W_q, weightSize);
        allocateCudaMemory(&W_k, weightSize);
        allocateCudaMemory(&W_v, weightSize);

        // Copy cpu data to device
        copyToCuda(W_q, W_q_tmp.data(), weightSize);
        copyToCuda(W_k, W_k_tmp.data(), weightSize);
        copyToCuda(W_v, W_v_tmp.data(), weightSize);
    }
    // TODO: add device="cpu" init case
}

float* AttentionLayer::forward(float *queries, float *keys, float *values){
    if(device == "cuda"){
        return selfAttention_gpu(queries, keys, values, W_q, W_k, W_v, seq_len, dim_emb);
    }
}