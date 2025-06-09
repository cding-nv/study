import torch
import intel_extension_for_pytorch
#import pdb

def simple_fn(x):
    for _ in range(20):
        y = torch.sin(x)
        x = x + y
    return x

#pdb.set_trace()
compiled_fn = torch.compile(simple_fn, backend="inductor")
input_tensor = torch.randn(10000)

r = compiled_fn(input_tensor)
print(r)
