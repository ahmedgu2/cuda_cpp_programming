#include <iostream>
#include <random>
#include <vector>
#include <chrono>

#define cudaErrorCheck(callReturn) do{ \
    cudaError_t error = callReturn; \
    if(error != cudaSuccess){ \
        std::cerr << "----CUDA ERROR in " << __FILE__ << " at line " << __LINE__ << "\n" << cudaGetErrorString(error) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}while(0)

#define cudaKernelErrorCheck() do{ \
    cudaError_t error = cudaGetLastError(); \
    if(error != cudaSuccess){ \
        std::cerr << "----CUDA ERROR in " << __FILE__ << " at line " << __LINE__ << "\n" << cudaGetErrorString(error) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}while(0);


std::vector<float> generateRandomVector(size_t n, float minValue, float maxValue){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(minValue, maxValue);
    
    std::vector<float> vector(n);
    for(int i = 0; i < n; ++i){
        vector[i] = dist(gen);
    }
    return vector;
}

__global__
void vectorAddition_gpu(float *d_v1, float *d_v2, float *output, size_t n){
    // grid-stride loop
    for(int indx = threadIdx.x + blockIdx.x * blockDim.x; indx < n; indx += gridDim.x * blockDim.x){
        output[indx] = d_v1[indx] + d_v2[indx];
    }
}

std::vector<float> vectorAddition_cpu(std::vector<float>& v1, std::vector<float>& v2){
    std::vector<float> output(v1.size());
    for(int i = 0; i < v1.size(); ++i)
        output[i] = v1[i] + v2[i];
    return output;
}


int main(){
    size_t N = 1e8;
    std::vector<float> v1 = generateRandomVector(N, 0.f, 5.f);
    std::vector<float> v2 = generateRandomVector(N, 0.f, 5.f);
    std::vector<float> gpuOutput(N);
    
    float *d_v1, *d_v2, *d_output;
    size_t sizeBytes = N * sizeof(float);
    // Init device arrays
    cudaErrorCheck(cudaMalloc((void**)&d_v1, sizeBytes));
    cudaErrorCheck(cudaMalloc((void**)&d_v2, sizeBytes));
    cudaErrorCheck(cudaMalloc((void**)&d_output, sizeBytes));

    // Copy host data to device
    cudaErrorCheck(cudaMemcpy(d_v1, v1.data(), sizeBytes, cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaMemcpy(d_v2, v2.data(), sizeBytes, cudaMemcpyHostToDevice));

    // Create events for timing the kernel
    cudaEvent_t startTime, endTime;
    cudaErrorCheck(cudaEventCreate(&startTime));
    cudaErrorCheck(cudaEventCreate(&endTime));

    // Run Kernel
    dim3 threadsPerBlock(512);
    dim3 numBlocks(1024);
    
    cudaErrorCheck(cudaEventRecord(startTime));
    vectorAddition_gpu<<<numBlocks, threadsPerBlock>>>(d_v1, d_v2, d_output, N);
    cudaErrorCheck(cudaEventRecord(endTime));
    cudaErrorCheck(cudaEventSynchronize(endTime));
    cudaKernelErrorCheck();
    
    // Calculate GPU time
    float gpuTimeMilliSeconds = 0.f;
    cudaErrorCheck(cudaEventElapsedTime(&gpuTimeMilliSeconds, startTime, endTime));

    // Copy result to host
    cudaErrorCheck(cudaMemcpy(gpuOutput.data(), d_output, sizeBytes, cudaMemcpyDeviceToHost));

    // CPU version for testing
    auto cpuStart = std::chrono::high_resolution_clock::now();
    auto cpuOutput = vectorAddition_cpu(v1, v2); 
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    auto cpuTimeMicroseconds = std::chrono::duration_cast<std::chrono::microseconds>(cpuEnd - cpuStart).count();

    // Print timing
    std::cout << "CPU function runtime: " << cpuTimeMicroseconds << " us" << std::endl;
    std::cout << "GPU kernel runtime: " << gpuTimeMilliSeconds * 1000 << " us" << std::endl;
    std::cout << "Speedup: " << cpuTimeMicroseconds / (gpuTimeMilliSeconds * 1000) << std::endl;

    // Check output matching
    std::cout << "Comparing outputs..." << std::endl;
    bool testFailed = false;
    for(int i = 0; i < v1.size(); ++i){
        if(gpuOutput[i] != cpuOutput[i]){
            std::cout << "Mismatch at index " << i << ": Expected " << cpuOutput[i] << ", Found " << gpuOutput[i] << std::endl;
            testFailed = true;
        }
    }
    if(testFailed)
        std::cout << "\033[31mTEST FAILED\033[0m" << std::endl;
    else
        std::cout << "\033[32mTEST PASSED!\033[0m" << std::endl;
    // test result
     
    cudaFree(d_v1);
    cudaFree(d_v2);
    cudaFree(d_output);
    return 0;
}