import torch

def dequant_block_fp8_weight_naive(weight, weight_scale, block_size, dtype=torch.bfloat16, original_M=None, original_N=None, do_unpad=False):
    if weight_scale is None:
        return weight
    assert len(block_size) == 2
    
    weight_shape_len = len(weight.shape)

    block_size_m, block_size_n = block_size

    # mul scale
    if weight_shape_len == 2:
        weight_scale_m, weight_scale_n = weight_scale.shape
        weight_scale = weight_scale.view(weight_scale_m, 1, weight_scale_n, 1)
        weight = weight.view(weight_scale_m, block_size_m, weight_scale_n, block_size_n)
        dequant_weight = weight.to(dtype) * weight_scale.to(dtype)
        dequant_weight = dequant_weight.view(weight_scale_m*block_size_m, weight_scale_n*block_size_n)
        keep_first_dim = False
    elif weight_shape_len == 3:
        fd, weight_scale_m, weight_scale_n = weight_scale.shape
        weight_scale = weight_scale.view(fd, weight_scale_m, 1, weight_scale_n, 1)
        weight = weight.view(fd, weight_scale_m, block_size_m, weight_scale_n, block_size_n)
        dequant_weight = weight.to(dtype) * weight_scale.to(dtype)
        dequant_weight = dequant_weight.view(fd, weight_scale_m*block_size_m, weight_scale_n*block_size_n)
        keep_first_dim = True
    else:
        raise ValueError("Only support original weight shape is either 2 or 3")

    return dequant_weight

import pdb
pdb.set_trace()
#qweight = torch.randint(low=0, high=100, size=(32, 7168, 2048), dtype=torch.int32)
#scales = torch.randn((32, 56, 16), dtype=torch.float32)
#block = [128, 128]
qweight = torch.randint(low=0, high=100, size=(2, 6, 4), dtype=torch.int32)

scales = torch.tensor([0.5, 1.0], dtype=torch.float32)[torch.randint(0, 2, (2, 3, 2))]

block = [2, 2]


weight_fp16 = dequant_block_fp8_weight_naive(qweight, scales, block)
print(weight_fp16.shape)
