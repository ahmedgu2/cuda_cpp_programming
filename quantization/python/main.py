import torch
import quantize_binding
import bitsandbytes.functional as F
import torch
import torch.utils.benchmark as benchmark
import nvtx

if __name__ == "__main__":
    torch.manual_seed(42)

    tensor = torch.randn(4096, 4096, dtype=torch.float16, device="cuda")
    print(quantize_binding.row_wise_quantization_gpu(tensor))
    print(F.int8_vectorwise_quant(tensor, threshold=0.1))