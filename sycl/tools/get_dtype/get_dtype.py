import torch
import torch.nn as nn
import torch.nn.functional as F

# class CustomModule(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.weight = nn.Parameter(torch.randn(3, 3))
#         self.bias = nn.Parameter(torch.zeros(3))

# model = CustomModule()

class SimpleAttention(nn.Module):
    def __init__(self, dim, dtype=torch.float32):
        super().__init__()
        self.query = nn.Linear(dim, dim).to(dtype)
        self.key = nn.Linear(dim, dim).to(dtype)
        self.value = nn.Linear(dim, dim).to(dtype)
        self.softmax = nn.Softmax(dim=-1) 

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        Q = Q.to(torch.bfloat16)
        K = K.to(torch.bfloat16)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.shape[-1] ** 0.5)
        attn_probs = self.softmax(attn_scores)  # Softmax
        V = V.to(torch.bfloat16)
        output = torch.matmul(attn_probs, V)
        return output

# Define Hook
def print_softmax_input_dtype(module, input):
    print(f"Softmax input dtype: {input[0].dtype}")

def print_softmax_output_dtype(module, input, output):
    print(f"Softmax output dtype: {output.dtype}")

# Create model and register Hook
dtype = torch.float8_e4m3fn  # Choose dtype
attn_model = SimpleAttention(dim=16, dtype=dtype)
attn_model.softmax.register_forward_pre_hook(print_softmax_input_dtype)
attn_model.softmax.register_forward_hook(print_softmax_output_dtype)

# Forward
x = torch.randn(2, 5, 16)  # 传入 bfloat16 data
x = x.to(torch.float8_e4m3fn)
output = attn_model(x)
print(f"Output dtype: {output.dtype}")

for name, module in attn_model.named_modules():
    if len(list(module.parameters())) > 0:  # Filter no parameters modules
        print(f"Layer: {name}, Type: {module.__class__.__name__}, Data Type: {next(module.parameters()).dtype}")
