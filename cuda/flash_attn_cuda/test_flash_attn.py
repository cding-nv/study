import torch
import flash_attn

torch.manual_seed(0)

B, T, H, D = 64, 64, 4, 64
q = torch.randn(B, T, H, D, dtype=torch.float16, device="cuda")
k = torch.randn_like(q)
v = torch.randn_like(q)
out = torch.zeros_like(q)

# 调用自定义 CUDA kernel
#flash_attn.flash_attn_forward(q, k, v, out)
flash_attn.flash_attn_blockwise_forward(q, k, v, out)

# 用标准 PyTorch 实现对比
q_ = q.transpose(1, 2)
k_ = k.transpose(1, 2)
v_ = v.transpose(1, 2)
scale = 1.0 / (D ** 0.5)
scores = torch.matmul(q_, k_.transpose(-2, -1)) * scale
probs = torch.softmax(scores, dim=-1)
out_torch = torch.matmul(probs, v_).transpose(1, 2)

diff = (out - out_torch).abs().max().item()
print(f"Max diff: {diff:.6f}")

