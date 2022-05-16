import time
import torch
from torch.utils.cpp_extension import load
nvcc_flags = ['-O3', '-std=c++14', '-use_fast_math', '-gencode', 'arch=compute_80,code=sm_80', '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__']
mod = load(name="mod", sources="mod.cu", extra_cuda_cflags=nvcc_flags, extra_include_paths=["../include"], verbose=True)


m, n, k = 256*128, 768, 3*768
# m, n, k = 5, 7, 11
alpha = 1.0
T = 321
use_fp16 = False


dtype = torch.float16 if use_fp16 else torch.float32
A  = torch.rand(m, k, dtype=dtype, device='cuda')
B  = torch.rand(k, n, dtype=dtype, device='cuda')
AT = torch.rand(k, m, dtype=dtype, device='cuda')
BT = torch.rand(n, k, dtype=dtype, device='cuda')


runTest = mod.runTest_fp16 if use_fp16 else mod.runTest_fp32
runTest(T, alpha, A, B, AT, BT);
