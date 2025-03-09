import torch
import quantize_binding

if __name__ == "__main__":
    batch = torch.rand(10, 5)
    result = quantize_binding.symmetric_quantization_gpu(batch)
    print(result)
    