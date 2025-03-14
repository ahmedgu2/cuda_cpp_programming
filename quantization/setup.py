from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="quantize_binding",
    ext_modules=[
        CUDAExtension(
            name="quantize_binding",                         # Name of the extension
            sources=[
                "bindings/quantize_binding.cpp",         # PyTorch binding code
                "src/symmetric.cu",          # CUDA kernel file
                "src/utils.cpp",            # Utility CUDA file
                "src/cuda_utils.cu",            # Utility CUDA file
                "src/LLM_int8.cu"
            ],
            include_dirs=["src"],            # Include directories for header files
            extra_compile_args={
                'cxx': ['-O3', '-Wall'],  # C++ compiler optimization flags
                'nvcc': ['-O3', '--use_fast_math']  # CUDA compiler flags (modify the architecture based on your GPU)
            }
        ),
    ],
    package_data={
        "quantize_binding": ["bindings/*.ipy"]
    },
    cmdclass={"build_ext": BuildExtension},  # Ensures the extension is built
)