import torch
import intel_extension_for_pytorch

def f(x):
    return torch.sin(x)**2 + torch.cos(x)**2

# First spot
compiled_f = torch.compile(f, backend='inductor', options={'trace.enabled':True, 'trace.graph_diagram':True})
input_tensor = torch.randn(10)

# second spot
output_from_compiled_f = compiled_f(input_tensor)
print(output_from_compiled_f)
