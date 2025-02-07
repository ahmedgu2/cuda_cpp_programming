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