# cudaKernels
Implementation of Cuda kernels.

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