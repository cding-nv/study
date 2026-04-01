import torch

torch.manual_seed(0)

# 参数
M = 8
K = 4
N = 5
G = 3   # num_groups

# 数据
A = torch.randint(0, 5, (M, K)).float()
B = torch.randint(0, 5, (G, N, K)).float()
m_indices = torch.randint(0, G, (M,))

print("A:")
print(A)
print("B:")
print(B)
print("m_indices:")
print(m_indices)

# 1. Naive 计算
C_naive = torch.zeros(M, N)

for i in range(M):
    g = m_indices[i]
    C_naive[i] = A[i] @ B[g].T

print("C_naive:")
print(C_naive)


# 2. Optimized grouped GEMM
C_opt = torch.zeros(M, N)

for g in range(G):
    # 找属于这个 group 的行
    rows = (m_indices == g).nonzero(as_tuple=True)[0] 
          # g=0, m_indices=[0, 0, 0, 1, 2, 0, 1, 1], -> rows = [0,1,2,5]
    if len(rows) == 0:
        continue

    A_g = A[rows]        # [Mg, K]
    B_g = B[g]           # [N, K]

    # GEMM
    C_g = A_g @ B_g.T    # [Mg, N]

    # scatter 回去
    C_opt[rows] = C_g

print("C_opt:")
print(C_opt)

# 对比
print("max diff:", (C_naive - C_opt).abs().max())



# 3. Batched version
# 为每一行选择对应的 B
B_selected = B[m_indices]      # [M, N, K] B repeat M times
A_expanded = A.unsqueeze(1)    # [M, 1, K]

C_batch = torch.bmm(A_expanded, B_selected.transpose(1, 2)).squeeze(1)

print("max diff vs naive:", (C_naive - C_batch).abs().max())

