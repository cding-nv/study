import torch

def compare_acc(result_fp32, result_bf16):
    result_bf16_to_fp32 = result_bf16.float()
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    cmp_cos = cos(result_fp32.reshape(-1), result_bf16.float().reshape(-1))
    cmp_v = cmp_cos.data.cpu().numpy()
    if cmp_v < 0.99:
        print(cmp_v)
        return cmp_v
    return -2.0

weights_bf16 = 
weights_fp32 = weights_bf16.float()

cmp_result = compare_acc(weights_fp32, weights_bf16)
if cmp_result != -2.0 and cmp_result != 0.0:
    file_path = str(cmp_result) + '.pt'
    torch.save(input_.cpu(), file_path)
    print("###### save ", file_path)

