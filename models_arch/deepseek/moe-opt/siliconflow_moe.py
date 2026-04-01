import torch
import torch.nn.functional as F

torch.manual_seed(0)

# -------------------------
# 参数
# -------------------------
tokens = 16
hidden = 32
ffn = 64
num_experts = 4
topk = 2
M_a = 4

# -------------------------
# 输入
# -------------------------
X = torch.randn(tokens, hidden)                #[16, 32]

W_gate = torch.randn(num_experts, hidden, ffn)    #[4, 32, 64]
W_up = torch.randn(num_experts, hidden, ffn)      #[4, 32, 64]
W_down = torch.randn(num_experts, ffn, hidden)    #[4, 64, 32]  

# router
topk_ids = torch.stack([
    torch.randperm(num_experts)[:topk]
    for _ in range(tokens)
])    #[16, 2]

topk_weights = torch.rand(tokens, topk)
topk_weights /= topk_weights.sum(dim=1, keepdim=True)  #[16, 2]

# --------------------------------------------------
# 1 naive MoE
# --------------------------------------------------
def moe_naive(X):
    Y = torch.zeros(tokens, hidden)
    for t in range(tokens):
        for k in range(topk):
            e = topk_ids[t, k]
            w = topk_weights[t, k]

            gate = F.silu(X[t] @ W_gate[e])   #[32, 64]
            up = X[t] @ W_up[e]
            h = gate * up
            out = h @ W_down[e]

            Y[t] += w * out

    return Y

# --------------------------------------------------
# 2 SiliconFlow style
# --------------------------------------------------
def moe_siliconflow(X):
    # -----------------
    # step1 regroup token by expert
    # -----------------
    expert_tokens = [[] for _ in range(num_experts)]
    expert_weights = [[] for _ in range(num_experts)]

    for t in range(tokens):
        for k in range(topk):
            e = int(topk_ids[t, k])
            expert_tokens[e].append(t)
            expert_weights[e].append(topk_weights[t, k])

    # -----------------
    # step2 padding
    # -----------------
    sorted_token_ids = []
    sorted_weights = []
    sorted_expert_ids = []

    for e in range(num_experts):
        ids = expert_tokens[e]
        ws = expert_weights[e]

        n = len(ids)
        pad = (M_a - n % M_a) % M_a

        ids = ids + [-1] * pad
        ws = ws + [0.0] * pad

        sorted_token_ids.extend(ids)
        sorted_weights.extend(ws)
        sorted_expert_ids.extend([e] * len(ids))

    sorted_token_ids = torch.tensor(sorted_token_ids)
    sorted_weights = torch.tensor(sorted_weights)
    sorted_expert_ids = torch.tensor(sorted_expert_ids)

    # -----------------
    # step3 scatter
    # -----------------
    valid_mask = sorted_token_ids >= 0
    X_scatter = torch.zeros(len(sorted_token_ids), hidden)
    X_scatter[valid_mask] = X[sorted_token_ids[valid_mask]]    # X_scatter [40, 32]
#(Pdb) print(topk_ids)
#tensor([[1, 2],
#        [3, 0],
#        [3, 1],
#        [0, 1],
#        [2, 0],
#        [2, 0],
#        [2, 1],
#        [1, 3],
#        [3, 0],
#        [0, 1],
#        [3, 1],
#        [1, 3],
#        [0, 2],
#        [0, 2],
#        [3, 0],
#        [3, 1]])
#(Pdb) print(sorted_token_ids)
#tensor([ 1,  3,  4,  5,  8,  9, 12, 13, 14, -1, -1, -1,  0,  2,  3,  6,  7,  9,
#        10, 11, 15, -1, -1, -1,  0,  4,  5,  6, 12, 13, -1, -1,  1,  2,  7,  8,
#        10, 11, 14, 15])
#(Pdb) print(sorted_expert_ids)
#tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3])
#(Pdb) print(valid_mask)
#tensor([ True,  True,  True,  True,  True,  True,  True,  True,  True, False,
#        False, False,  True,  True,  True,  True,  True,  True,  True,  True,
#         True, False, False, False,  True,  True,  True,  True,  True,  True,
#        False, False,  True,  True,  True,  True,  True,  True,  True,  True])
# (Pdb) print(sorted_token_ids[valid_mask])
# tensor([ 1,  3,  4,  5,  8,  9, 12, 13, 14,  0,  2,  3,  6,  7,  9, 10, 11, 15,
#         0,  4,  5,  6, 12, 13,  1,  2,  7,  8, 10, 11, 14, 15])
# X_scatter[0] = X[1]
# X_scatter[1] = X[3]
# X_scatter[2] = X[4]
# ......


    # -----------------
    # step4 scatter_gemm
    # -----------------
    out_scatter = torch.zeros(len(sorted_token_ids), hidden)
    for i in range(0, len(sorted_token_ids), M_a):   # i 每次+M_a
        e = sorted_expert_ids[i]
        block = X_scatter[i:i+M_a]
        gate = F.silu(block @ W_gate[e])
        up = block @ W_up[e]
        h = gate * up

        out = h @ W_down[e]
        out_scatter[i:i+M_a] = out

    # -----------------
    # step5 gather
    # -----------------
    Y = torch.zeros(tokens, hidden)
    idx = 0
    for e in range(num_experts):
        ids = expert_tokens[e]
        for t_i in ids:
            w = sorted_weights[idx]
            Y[t_i] += w * out_scatter[idx]
            idx += 1
        pad = (M_a - len(ids) % M_a) % M_a
        idx += pad
    return Y

# --------------------------------------------------
# 运行
# --------------------------------------------------

Y_naive = moe_naive(X)
Y_sf = moe_siliconflow(X)

print("max diff =", (Y_naive - Y_sf).abs().max())