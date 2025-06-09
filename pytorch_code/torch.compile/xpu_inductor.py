import torch
import intel_extension_for_pytorch

def simple_fn(x):
    for _ in range(20):
        y = torch.sin(x).to('xpu')
        x = x + y
    return x

compiled_fn = torch.compile(simple_fn, backend="inductor")
input_tensor = torch.randn(10000).to(device="xpu")

r = compiled_fn(input_tensor)
print(r)
