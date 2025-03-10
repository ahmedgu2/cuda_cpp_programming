#pragma once
#include <iostream>

/******************* Utility macros and functions ********************** */
#define CUDA_CHECK_ERROR(callResult)                                                \
    do                                                                              \
    {                                                                               \
        cudaError_t error = callResult;                                             \
        if (error != cudaSuccess)                                                   \
        {                                                                           \
            std::cout << "Error in " << __FILE__ << " at line " << __LINE__ << "\n" \
                      << cudaGetErrorString(error);                                 \
            exit(EXIT_FAILURE);                                                     \
        }                                                                           \
    } while (0)

#define CUDA_KERNEL_CHECK_ERROR()                                                            \
    do                                                                                       \
    {                                                                                        \
        cudaError_t error = cudaGetLastError();                                              \
        if (error != cudaSuccess)                                                            \
        {                                                                                    \
            std::cerr << "----CUDA ERROR in " << __FILE__ << " at line " << __LINE__ << "\n" \
                      << cudaGetErrorString(error) << std::endl;                             \
            exit(EXIT_FAILURE);                                                              \
        }                                                                                    \
    } while (0)

#define LOG(msg) std::cout << msg << std::endl