
__global__
void max_(float *array, size_t length, float* blockMax){
    extern __shared__ float max_s[];
    int tx = threadIdx.x; 
    int segment = 2 * blockDim.x; 
    int indx = tx + segment * blockIdx.x;

    if(indx < length)
        max_s[tx] = array[indx];
    if(indx + blockDim.x < length)
        max_s[tx] = fmax(max_s[tx], array[indx + blockDim.x]);

    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1){
        __syncthreads();
        if(tx < stride)
            max_s[tx] = fmax(max_s[tx], max_s[tx + stride]);
    }
    if(tx == 0){
        blockMax[blockIdx.x] = max_s[tx];
    }
}

__global__
void min_(float *array, size_t length, float* blockMin){
    extern __shared__ float min_s[];
    int tx = threadIdx.x; 
    int segment = 2 * blockDim.x; 
    int indx = tx + segment * blockIdx.x;

    if(indx < length)
        min_s[tx] = array[indx];
    if(indx + blockDim.x < length)
        min_s[tx] = fmin(min_s[tx], array[indx + blockDim.x]);

    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1){
        __syncthreads();
        if(tx < stride)
            min_s[tx] = fmin(min_s[tx], min_s[tx + stride]);
    }
    if(tx == 0){
        blockMin[blockIdx.x] = min_s[tx];
    }
}
