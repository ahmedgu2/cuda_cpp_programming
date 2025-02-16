# CUDA C/C++ Programming
This is a 100 days challenge to master CUDA C/C++ programming. 
I'll be also learning new C++ language features (C++ 14/17/20) and experimenting with them when possible.

### Resources:
- Programming Massively Parallel Processors 4th edition by David Kirk and Wen-mei Hwu.
- [Oak Ridge x Nvidia CUDA Training series](https://www.olcf.ornl.gov/cuda-training-series/)

# Progress
### Day 1 
- Refresher on basic cuda syntax and notions.
- Read PMPP Chapter 4.
- Implemented simple **vector addition** using **grid-stride loop**.
- Refresher on C++ vectors and timing (std::chrono).
- Learned about `cudaEvent_t`, `cudaEventCreate(&event)`, `cudaEventRecord(&event)` and `cudaEventTimeElapsed(&ms, start, end)` and how to use them for computing the runtime of a cuda kernel.
- Benchmarked CPU vs GPU versions.

### Day 2
- Started reading Chapter 5 as a refresher for different types of device memory.
- Implemented **naive matrix multiplication**.
- Refresher on C++ 2d arrays and dynamic memory allocation.
- Benchmarked CPU vs GPU versions.

### Day 3
- Profiled the naive matrix multiplication (Nsight compute) and reduced runtime by 50% by tuning the number of threads per block (reduced threadsPerBlock from 32x32 to 16x16 and increased the number of blocks from 52 to 108 for better balancing of workload in SMs).
- Implemented a small script to get the device propreties.
- Finished reading PMPP Chapter 5.
- Implemented tiled **matrix multiplication with shared memory**.
- Benchmarked `mmSharedTile_gpu` with the CPU and naive GPU version.

### Day 4:
- Re-read some parts of Chapter 4-5.
- Modified `mmSharedTile_gpu()` to handle arbitrary matrix width (`N` does have to be divsible by `TILE_DIM`).

### Day 5:
- Worked on some of the exercies of Chapter 5.
- Started reading Chapter 6.
- Modified `mmSharedTile_gpu()` to handle arbitrary matrix size (nRows != nCols)

### Day 6:
- Continued reading Chapter 6.
- Started reading about gaussian blur and implementing it.
- Implemented the 2D version of normal distribution density function `normal_pdf_gpu()` and `normal_pdf_cpu()` 

### Day 7: 
- Finished implementing gaussian blur on cpu. Used the "edge expension" padding strategy.
- Created some testing code to compare outputs with `cv2` outputs (in python).
- Refresher on `fstream` in C++.

### Day 8:
- Implemented a naive version of gaussian blur on gpu.
- Had a refresher on `atomicAdd()` and `cudaDeviceSynchronize()`.
- Something weird: Random init using `std::mt19937(seed)` yielded different results when I compile .cu and .cpp files containing the same init function. Is this due to different compilers / flags used for g++ vs nvcc? Need to investigate more.

### Day 9:
- Read Chapter 7 about convolutions.
- Optimized the gaussian blur by storing the kernel in constant memory for efficient use of cache and higher performance.
- Started incorporating the tiling technique with shared memory to address memory bandwith bottlenecks.

### Day 10:
- Finished reading Chapter 7.
- Finished Optimizing gaussian blur using tiling and shared memory.

### Day 11:
- Implemented some activation functions for 2D matrices: relu, sigmoid and tanh.
- Re-read Chapter 6 and 7.

### Day 12: 
- Started reading Chapter 10 about redaction patterns.
- Implemented simple sum redaction `simpleSumReduction_gpu()` that handles arrays that fit only in 1 block.
- Implemented `simpleSumReductionV2_gpu()` which uses a better thread assignement strategy for better execution resource utlization and less control divergence.

### Day 13:
- Finished reading Chapter 10.
- Optimized sum reduction by implementing `sumReductionSharedMultiSegment_gpu()` which uses shared memory for load time reduction and generlizes to arbitrary sized arrays (i.e. not limited by 1 block).