# import all necessary libraries
import torch
from torch.profiler import profile, ProfilerActivity
import intel_extension_for_pytorch

# these lines won't be profiled before enabling profiler tool
input_tensor = torch.randn(1024, dtype=torch.float32, device='xpu:0')

# enable Kineto supported profiler tool with a `with` statement
with profile(activities=[ProfilerActivity.CPU,
                         ProfilerActivity.XPU]) as prof:
    # do what you want to profile here after the `with` statement with proper indent
    output_tensor_1 = torch.nonzero(input_tensor)
    output_tensor_2 = torch.unique(input_tensor)

# print the result table formatted by the profiler tool as your wish
print(prof.key_averages().table())
