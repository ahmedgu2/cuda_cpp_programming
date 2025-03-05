#pragma once
#include <vector>
#include <string>

class AttentionLayer{
public:
    AttentionLayer(uint32_t seq_len, uint32_t dim_len, std::string device="cuda");
    ~AttentionLayer();

    float* forward(float *queries, float *keys, float *values);

private:
    // Weight matrices
    float *W_q, *W_k, *W_v;
    uint32_t seq_len, dim_emb;
    std::string device;

    void initWeights();

};