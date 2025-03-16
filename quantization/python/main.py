import torch
import quantize_binding
import bitsandbytes.functional as F
import torch
import torch.utils.benchmark as benchmark
import nvtx

if __name__ == "__main__":
    torch.manual_seed(42)

    @nvtx.annotate("bitsandbytes_impl_noTransfer")
    def bnb_noTransfer(tensor: torch.Tensor):
        return F.int8_vectorwise_quant(tensor)
        
    @nvtx.annotate("my_impl")
    def custom(tensor: torch.Tensor):
        return quantize_binding.row_wise_quantization_gpu(tensor)

    def warmup(tensor, steps=1):
        q_tensor = quantize_binding.row_wise_quantization_gpu(tensor)
        q_tensor_bnb = F.int8_vectorwise_quant(tensor)
        torch.cuda.synchronize()


    tensor = torch.randn(4096, 4096, dtype=torch.float16, device="cuda")
    warmup(tensor, 5)

    for i in range(50):
        torch.cuda.synchronize()
        custom_result = custom(tensor)
        torch.cuda.synchronize()
        bnb_result = bnb_noTransfer(tensor)
    