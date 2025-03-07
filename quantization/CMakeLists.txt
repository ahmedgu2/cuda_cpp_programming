cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(attention LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_DEBUG "/Zi /Od")
set(CMAKE_BUILD_TYPE Debug)

# Find CUDA
enable_language(CUDA)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# Source files
set(
    CUDA_SOURCES
    src/symmetric.cu
)

set(
    CPP_SOURCES
    src/utils.cpp
    src/quantization_cpu.cpp
)

# Create a library from cuda files
add_library(cuda_kernels_lib ${CPP_SOURCES} ${CUDA_SOURCES})
set_target_properties(cuda_kernels_lib PROPERTIES
    # CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Add test executable
add_executable(run_tests test/test.cpp)
target_link_libraries(run_tests PRIVATE cuda_kernels_lib ${CUDA_LIBRARIES})