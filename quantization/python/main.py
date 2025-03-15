import torch
import quantize_binding
import bitsandbytes.functional as F
import torch
import torch.utils.benchmark as benchmark
import nvtx

if __name__ == "__main__":
    torch.manual_seed(42)

    @nvtx.annotate("bitsandbytes_impl")
    def bnb(tensor: torch.Tensor):
        # Your PyTorch operations here
        tensor = tensor.to("cuda")
        q_tensor_bnb = F.int8_vectorwise_quant(tensor)
        q_tensor_bnb[0].to("cpu")
        torch.cuda.synchronize()

    @nvtx.annotate("bitsandbytes_impl_noTransfer")
    def bnb_noTransfer(tensor: torch.Tensor):
        # Your PyTorch operations here
        # tensor = tensor.to("cuda")
        q_tensor_bnb = F.int8_vectorwise_quant(tensor)
        # q_tensor_bnb[0].to("cpu")
        torch.cuda.synchronize()
        
    @nvtx.annotate("my_impl")
    def custom(tensor: torch.Tensor):
        q_tensor = quantize_binding.row_wise_quantization_gpu(tensor)

    def warmup(tensor, steps=1):
        q_tensor = quantize_binding.row_wise_quantization_gpu(tensor)
        tensor = tensor.to("cuda")
        q_tensor_bnb = F.int8_vectorwise_quant(tensor)
        q_tensor_bnb[0].to("cpu")
        torch.cuda.synchronize()


    tensor = torch.randn(4096, 4096)
    tensor = tensor.type(torch.float16)
    warmup(tensor, 2)

    custom(tensor)
    bnb(tensor)
    tensor = tensor.to("cuda")
    bnb_noTransfer(tensor)
    # t = benchmark.Timer(stmt="bnb(tensor)", globals={"bnb": bnb, "tensor": tensor})
    # print(t.timeit(100))  # Runs 100 times and averages the result

    # t = benchmark.Timer(stmt="custom(tensor)", globals={"custom": custom, "tensor": tensor})
    # print(t.timeit(100))  # Runs 100 times and averages the result

# import torch
# import quantize_binding
# import bitsandbytes.functional as F
# import torch.utils.benchmark as benchmark
# import torch.profiler

# import torch
# import quantize_binding
# import bitsandbytes.functional as F
# import torch.profiler

# if __name__ == "__main__":
#     torch.manual_seed(42)

#     def bnb(tensor: torch.Tensor):
#         tensor = tensor.to("cuda")
#         q_tensor_bnb = F.int8_vectorwise_quant(tensor)
#         q_tensor_bnb[0].to("cpu")

#     def custom(tensor: torch.Tensor):
#         q_tensor = quantize_binding.row_wise_quantization_gpu(tensor)

#     tensor = torch.randn(4096, 4096, dtype=torch.float16)

#     with torch.profiler.profile(
#         activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
#         schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
#         on_trace_ready=torch.profiler.tensorboard_trace_handler("./bnb_profiler"),
#         record_shapes=True,
#         profile_memory=True,
#         with_stack=True
#     ) as prof:
#         for _ in range(10):  # Run multiple iterations for better profiling
#             bnb(tensor)
#             prof.step()  # Notify profiler of step

#     with torch.profiler.profile(
#         activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
#         schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
#         on_trace_ready=torch.profiler.tensorboard_trace_handler("./custom_profiler"),
#         record_shapes=True,
#         profile_memory=True,
#         with_stack=True
#     ) as prof:
#         for _ in range(5):
#             custom(tensor)
#             prof.step()

#     print("Profiling completed! You can now visualize the results using TensorBoard.")

