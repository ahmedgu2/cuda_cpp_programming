from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="quantize_cuda",
    ext_modules=[
        CUDAExtension(
            name="quantize_cuda",                         # Name of the extension
            sources=[
                "bindings/quantize_binding.cpp",         # PyTorch binding code
                "src/symmetric.cu",          # CUDA kernel file
                "src/utils.cpp",            # Utility CUDA file
                "src/cuda_utils.cu",            # Utility CUDA file
            ],
            include_dirs=["src"]            # Include directories for header files
        ),
    ],
    cmdclass={"build_ext": BuildExtension},  # Ensures the extension is built
)