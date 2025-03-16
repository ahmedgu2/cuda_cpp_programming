# CUDA C/C++ Programming
This is a 100 days challenge to master CUDA C/C++ programming. 
I'll be also learning new C++ language features (C++ 14/17/20) and experimenting with them when possible.

### Resources:
- Programming Massively Parallel Processors 4th edition by David Kirk and Wen-mei Hwu.
- [Oak Ridge x Nvidia CUDA Training series](https://www.olcf.ornl.gov/cuda-training-series/)

# Progress
## Day 1 
- Refresher on basic cuda syntax and notions.
- Read PMPP Chapter 4.
- Implemented simple **vector addition** using **grid-stride loop**.
- Refresher on C++ vectors and timing (std::chrono).
- Learned about `cudaEvent_t`, `cudaEventCreate(&event)`, `cudaEventRecord(&event)` and `cudaEventTimeElapsed(&ms, start, end)` and how to use them for computing the runtime of a cuda kernel.
- Benchmarked CPU vs GPU versions.

## Day 2
- Started reading Chapter 5 as a refresher for different types of device memory.
- Implemented **naive matrix multiplication**.
- Refresher on C++ 2d arrays and dynamic memory allocation.
- Benchmarked CPU vs GPU versions.

## Day 3
- Profiled the naive matrix multiplication (Nsight compute) and reduced runtime by 50% by tuning the number of threads per block (reduced threadsPerBlock from 32x32 to 16x16 and increased the number of blocks from 52 to 108 for better balancing of workload in SMs).
- Implemented a small script to get the device propreties.
- Finished reading PMPP Chapter 5.
- Implemented tiled **matrix multiplication with shared memory**.
- Benchmarked `mmSharedTile_gpu` with the CPU and naive GPU version.

## Day 4:
- Re-read some parts of Chapter 4-5.
- Modified `mmSharedTile_gpu()` to handle arbitrary matrix width (`N` does have to be divsible by `TILE_DIM`).

## Day 5:
- Worked on some of the exercies of Chapter 5.
- Started reading Chapter 6.
- Modified `mmSharedTile_gpu()` to handle arbitrary matrix size (nRows != nCols)

## Day 6:
- Continued reading Chapter 6.
- Started reading about gaussian blur and implementing it.
- Implemented the 2D version of normal distribution density function `normal_pdf_gpu()` and `normal_pdf_cpu()` 

## Day 7: 
- Finished implementing gaussian blur on cpu. Used the "edge expension" padding strategy.
- Created some testing code to compare outputs with `cv2` outputs (in python).
- Refresher on `fstream` in C++.

## Day 8:
- Implemented a naive version of gaussian blur on gpu.
- Had a refresher on `atomicAdd()` and `cudaDeviceSynchronize()`.
- Something weird: Random init using `std::mt19937(seed)` yielded different results when I compile .cu and .cpp files containing the same init function. Is this due to different compilers / flags used for g++ vs nvcc? Need to investigate more.

## Day 9:
- Read Chapter 7 about convolutions.
- Optimized the gaussian blur by storing the kernel in constant memory for efficient use of cache and higher performance.
- Started incorporating the tiling technique with shared memory to address memory bandwith bottlenecks.

## Day 10:
- Finished reading Chapter 7.
- Finished Optimizing gaussian blur using tiling and shared memory.

## Day 11:
- Implemented some activation functions for 2D matrices: relu, sigmoid and tanh.
- Re-read Chapter 6 and 7.

## Day 12: 
- Started reading Chapter 10 about redaction patterns.
- Implemented simple sum redaction `simpleSumReduction_gpu()` that handles arrays that fit only in 1 block.
- Implemented `simpleSumReductionV2_gpu()` which uses a better thread assignement strategy for better execution resource utlization and less control divergence.

## Day 13:
- Finished reading Chapter 10.
- Optimized sum reduction by implementing `sumReductionSharedMultiSegment_gpu()` which uses shared memory for load time reduction and generlizes to arbitrary sized arrays (i.e. not limited by 1 block).

## DAY 14:
- Read about precision loss for parallel reduction sum. Floating point sum yields different results depending on the algorithm, e.g. sequential sum (cpu version) results != Hierarchical reduction sum (gpu version) results.
- Added an integer version of the kernel which yields the exact same result as the cpu version.

## Day 15:
- Started to work on implementing **Softmax**.
- Got a refresher about the log-sum-exp and maximum tricks for numerical stability of softmax. 
- Implemented **1D Parallel maximum**.

## Day 16:
- Implemented **Softmax** function as described below:

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}$$


- To improve numerical stability, we use the **max trick**, where we subtract the maximum value from all inputs:

$$ \text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_{j} e^{x_j - \max(x)}} $$
- The softmax function was implemented in a non-optimized manner, with each operation (subtraction, exponentiation, division, max, and sum) executed in separate kernels. This introduces overhead and increases runtime.
- The next step is to apply kernel fusion to enhance performance.

## Day 17:
- Optimized the softmax function by applying kernel fusion as follows:
    - `expArray_gpu()`, `substractArray_gpu()` are now combined and integrated into the `sumArray_gpu()` kernel (named `sumExpArray_gpu()`).
 - This helped to improve performance by:
    - **Reduction of kernel launch overhead** as we reduced the number of kernel from 5 to 3.
    - **Lower Memory Bandwidth Usage** as global memory reads and writes (i.e. less intermediate results) are decreased in the fused kernels.
    - **Better Memory Locality**.
- Results:
    - Improved runtime by ***30%*** on a test 1D array of size 4096 x 1024.

## Day 18:
- Read Chapter 9 (Histograms).
- Implemented naive parallel histogram kernel.

## Day 19:
- I'll be implementing different types of attention for the next few days, e.g. scoring functions such as **dot product**, **Bahdanu**, **Luong**, **self attention**..
- Refresher on attention in general, and Bahdanau and Luong scoring functions.
- Implemented the simple **dot product** based scoring function.

## Day 20:
- Very short session (not enough time to work on much).
- Fused dot product kernels (multiplication and sum) into one for better performance and less memory consomption.
- Layed the idea for implementing **Luong attention** (will implement tomorrow).

## Day 21:
- Implemented **Luong attention** scoring function, i.e. **Bilinear attention score**:

$$
\text{score}(h_t, s_k) = h^T_{t} \times W \times s_k
$$

- We can compute all scores $(s_k)$ for $k$ in $1..m$ in one go as follows:

$$
\text{scores}(h_t, S) = h^T_{t} \times W \times S
$$

## Definitions:
- `S`: Encoder states matrix of shape `(d, m)`, where each column represents an encoder state $s_k$, for $k$ in $1..m$.
- `W`: Weight matrix of shape `(d, d)`.
- `h^T_{t}`: The decoder vector at timestep $t$ of shape `(1, d)`.

## Day 22:
- Quick session, started implementing **Bahdanau attention** score, i.e **MLP score**
$$
\text{score}(h_t, s_k) = W^T_{2} \times tanh (W_1 [h, s_k])
$$

## Day 23:
- Completed the **Bahdanau attention score** implmentation.
- Implemented **vector concatination** and 2D **tanh** which are used to calculate the score.

## Day 24:
- Refresher on CMake and how to create one for cuda and cpp project.
- Refactored some of the attention to seperate kernels from testing code.

## Day 25:
- Started implementing **self attention**
- Implemented **2D Softmax** kernel and run testing for cpu vs gpu versions.
- Contrary to my 1D Softmax implementation which used 3 kernels, the 2D version uses only 1 kernel which gets rid of the kernel calls overhead.

## Day 26:
- Implemented utility functions for self attention:
    - `matmul()` is a tiled matrix multiplication with shared memory that handles arbitrary matrix shapes.
    - `matmulScaledTranspose()` computes $\frac{Q K^T}{scalingFactor}$.
- Implemented **Self Attention** which combines the `softmax2D()`, `matmul()` and `matmulScaledTranspose()` to get the attention matrix given $Q$, $K$, and $V$.
- Next up is to test it and then refactor it into a *AttentionLayer* class to seperate the weights creation and intialization from forward pass. The next objective would be to  align it with Pytorch's `MultiHeadAttention()` layer and compare performance (using only 1 head for pytorch's layer as a first step).


## Day 27:
- Developed the `AttentionLayer` class to replicate PyTorch's `MultiHeadAttention` functionality.
- Refactored the `selfAttention()` kernel to seamlessly integrate with `AttentionLayer`.
- Designed `AttentionLayer` to handle data allocation and initialization using Xavier normal initialization on a CUDA device.
- Implemented the `forward()` function to invoke the self-attention kernel and return the computed attention output.

Example usage:
```cpp
    AttentionLayer attentionLayer(seq_len, dim_emb, "cuda");
    float* attentionOuput = attentionLayer.forward(queries, keys, values);
```

## Day 28:
- Implemented the CPU version of selfAttention.
- Added testing for GPU version and spent some time debugging.
- Fixed boundary checks and indexing bugs in the `matmulTransposScaled()` kernel.

## Day 29:
- In the next few days, I'll be implementing different kind of quantization techniques (symmetric, asymmetric, LLM.int8, ...)
- Implemented **Symmetric quantization**.
#### Definitions:

1. **Scale factor (S)**:

$$
S = \frac{\max(|x|)}{2^{b-1} - 1}
$$

   where:
   - $\max(|x|)$ is the maximum floating-point value.
   - $b$ is the bit width of the quantized representation.

2. **Zero-point (Z)**:  
   - In symmetric quantization:  

$$
Z = 0
$$

### Quantization and Dequantization:

- **Quantization:**

$$
q = \text{round}\left(\frac{x}{S}\right)
$$

  where $x$ is the floating-point value, and $q$ is the quantized integer.

- **Dequantization:**

$$
x = q \cdot S
$$

## DAY 30:
- Implemented cpu version for 8bit symmetric quantization for testing vs gpu one and fixed a bug in max calculations for the gpu version.
- Implemented **8 bits Asymmetric quantization** (gpu and cpu version + testing):
#### Definitions:

1. **Scale factor (S)**:  

$$
S = \frac{x_{\max} - x_{\min}}{q_{\max} - q_{\min}}
$$

   where:  
   - $x_{\max}$, $x_{\min}$ are the maximum and minimum floating-point values.  
   - $q_{\max}$, $q_{\min}$ are the maximum and minimum quantized integer values.  

2. **Zero-point (Z)**:  
   - In asymmetric quantization:  
   
$$
Z = \text{round} \left( q_{\min} - \frac{x_{\min}}{S} \right)
$$

### Quantization and Dequantization:

- **Quantization:**

$$
q = \text{round} \left( \frac{x}{S} + Z \right)
$$
  
  where $x$ is the floating-point value, and $q$ is the quantized integer.

- **Dequantization:**

$$
x = (q - Z) \cdot S
$$

## Day 31:
- Learned how to add python binding for cuda kernels and spent some time debugging some windows errors. 
- Added python binding test for symmetric quantization.

## Day 32:
- Read the **LLM.int8()** paper.
- Started implementing the LLM.int8() specific kernels.
- Implemented **vector-wise quantization** which applies 8 bits absmax quantization row-wise (will implement column-wise later).
- Implemented the CPU version and tested my kernel against it.
- Next up will be implementing the mixed-precision multiplication for outliers / non-outliers matrices.

## Day 33:
- Continued implementing LLM.int8().
- Implemented gpu and cpu version of **column-wise quantization** and tested it.
- Implemented the third step of LLM.int8() which is basically a matmul between 2 *int8* which results in a *int32* matrix.

## Day 34:
- Finished working on the building blocks of **LLM.int8()** quantization method.
- Implemented **tiled matrix multiplication** and learned how to dynamically allocate shared memory inside templated functions.
- Implemented **dequentization** kernel.
- Implemented **outlier detection** kernel to get the columns that contain at least 1 value higher than `threshold` (defaults to 6).
- Implemented cpu versions + basic testing for all the above kernels.

## Day 35:
- Converted `rowWiseQuant8bits()` to work with half precision (fp16) and refactored the testing function to handle it.
- Added pytorch binding for that function to allow testing vs pytorch version. 
- Learned more about `torch.profiler` and run benchmarks against *bitsandbytes* `int8_vectorwise_quant()` function.
- My cuda implementation is 30% slower: 8.59ms vs 11.6ms. Note that my cuda kernel (i.e. computation) runs in 406us, which means a lot of the time is spent copying data between host and device.
- I'll dig deeper into profling to see what bottlenecks my implementation has and how to improve it.

## Day 36:
- Focused on profiling C++ and Python-binded CUDA kernels using Nvidia Nsight Systems and NVTX.
- Identified bottlenecks in vector-wise quantization, primarily due to:
   - Unnecessary memory allocation which leads to more overhead for copying data between CPU and GPU freeing memory.
- Wrapper function around the kernel is 30% slower due to these issues.
- However, the actual kernel execution time is 39µs, faster than BitsandBytes (54µs).
- BitsandBytes includes an extra outlier detection step, which accounts for some of the difference.

## Day 37:
- Detected my implementation's bottlenecks and optimized it:
   - My kernel wrapper was taking host pointers (from `torch::Tensor.data_ptr<>(tensor)`) and then allocating, copying and freeing new device arrays which added most of the overhead, wheras bnb implementation takes
   tensors already on device.
      
      => Modified my kernel wrapper to directly take device pointers from torch tensors.
   - I was using `torch::zeros()` to init tensors that will contain the results, which is inefficiant as it has an extra uncessary init operation. 
   
      => Replaced it with `torch::empty()`.
- Created more robust benchmarking:
  - 50 runs per function to get more reliable statistics.
  - Use `torch.cuda.synchronize()` before and after function calls to ensure device readiness.
### Results:
  Benchmark was done on a **4096x4096** tensor.
  - *My implementation*: ***34us***.
  - *Bitsandbytes implementation*: ***53us***.

Note that bitsandbytes implementation contains 1 extra step for detecting the outliers.