# fmha v2

This is to simplify [TensorRT-LLM fmha_v2](https://github.com/NVIDIA/TensorRT-LLM/tree/main/cpp/kernels/fmha_v2) for cuda-gdb and kernel debugging.
Platform: GeForce RTX 4060 / SM89

## 1. Build
### Build fmha_v2_flash_attention_fp16_128_128_S_qkv_16_sm89.cu
```
$ nvcc  -O3 -std=c++17 -ccbin g++ -use_fast_math -Xptxas=-v --expt-relaxed-constexpr -g -lineinfo -DUSE_SAME_SUM_ORDER_IN_SOFTMAX_AS_REF_CODE -DHALF_ACCUMULATION_FOR_FLASH_ATTENTION  -DUSE_I2F_EMULATION_TRICK -DUSE_F2I_EMULATION_TRICK -gencode=arch=compute_89,code=\"sm_89\" -I./ -I./generated -I/usr/local/cuda/include -c -o fmha_v2_flash_attention_fp16_128_128_S_qkv_16_sm89.cu.o generated/fmha_v2_flash_attention_fp16_128_128_S_qkv_16_sm89.cu
```
```
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function 'fmha_v2_flash_attention_fp16_128_128_S_qkv_16_custom_mask_sm89_kernel_nl_tiled' for 'sm_89'
ptxas info    : Function properties for fmha_v2_flash_attention_fp16_128_128_S_qkv_16_custom_mask_sm89_kernel_nl_tiled
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 203 registers, 1024 bytes cmem[0]
ptxas info    : Compiling entry function 'fmha_v2_flash_attention_fp16_128_128_S_qkv_16_sliding_or_chunked_causal_sm89_kernel_nl_tiled' for 'sm_89'
ptxas info    : Function properties for fmha_v2_flash_attention_fp16_128_128_S_qkv_16_sliding_or_chunked_causal_sm89_kernel_nl_tiled
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 251 registers, 1024 bytes cmem[0]
ptxas info    : Compiling entry function 'fmha_v2_flash_attention_fp16_128_128_S_qkv_16_causal_sm89_kernel_nl_tiled' for 'sm_89'
ptxas info    : Function properties for fmha_v2_flash_attention_fp16_128_128_S_qkv_16_causal_sm89_kernel_nl_tiled
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 245 registers, 1024 bytes cmem[0]
ptxas info    : Compiling entry function 'fmha_v2_flash_attention_fp16_128_128_S_qkv_16_sm89_kernel_nl_tiled' for 'sm_89'
ptxas info    : Function properties for fmha_v2_flash_attention_fp16_128_128_S_qkv_16_sm89_kernel_nl_tiled
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 229 registers, 1024 bytes cmem[0]
In file included from tmpxft_00002156_00000000-6_fmha_v2_flash_attention_fp16_128_128_S_qkv_16_sm89.cudafe1.stub.c:1:
generated/fmha_v2_flash_attention_fp16_128_128_S_qkv_16_sm89.cu: In function ‘void fmha_v2_flash_attention_fp16_128_128_S_qkv_16_sm89_kernel_nl_tiled(_ZN4bert35Fused_mult
ihead_attention_params_v2E)’:
generated/fmha_v2_flash_attention_fp16_128_128_S_qkv_16_sm89.cu:295:6: note: the ABI for passing parameters with 64-byte alignment has changed in GCC 4.6
  295 | void fmha_v2_flash_attention_fp16_128_128_S_qkv_16_sm89_kernel_nl_tiled(bert::Fused_multihead_attention_params_v2 params){
      |      ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```

### Build softmax_fp16.cu
```
$ nvcc  -O3 -std=c++17 -ccbin g++ -use_fast_math -Xptxas=-v --expt-relaxed-constexpr -g -lineinfo -DUSE_SAME_SUM_ORDER_IN_SOFTMAX_AS_REF_CODE -DHALF_ACCUMULATION_FOR_FLASH_ATTENTION  -I./ -I./generated -I/usr/local/cuda/include -gencode=arch=compute_89,code=\"sm_89\" -c -o softmax_fp16.cu.o softmax_fp16.cu
```
```
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z14softmax_kernelIttLi65536ELi4ELi4EEv14Softmax_paramsIT_T0_E' for 'sm_89'
ptxas info    : Function properties for _Z14softmax_kernelIttLi65536ELi4ELi4EEv14Softmax_paramsIT_T0_E
    17296 bytes stack frame, 35396 bytes spill stores, 32256 bytes spill loads
ptxas info    : Used 255 registers, 456 bytes cmem[0]
ptxas info    : Compiling entry function '_Z14softmax_kernelIttLi32768ELi4ELi4EEv14Softmax_paramsIT_T0_E' for 'sm_89'
ptxas info    : Function properties for _Z14softmax_kernelIttLi32768ELi4ELi4EEv14Softmax_paramsIT_T0_E
    9520 bytes stack frame, 22740 bytes spill stores, 32384 bytes spill loads
ptxas info    : Used 255 registers, 456 bytes cmem[0]
ptxas info    : Compiling entry function '_Z14softmax_kernelIttLi16384ELi4ELi4EEv14Softmax_paramsIT_T0_E' for 'sm_89'
ptxas info    : Function properties for _Z14softmax_kernelIttLi16384ELi4ELi4EEv14Softmax_paramsIT_T0_E
    6192 bytes stack frame, 12704 bytes spill stores, 17632 bytes spill loads
ptxas info    : Used 255 registers, 456 bytes cmem[0]
ptxas info    : Compiling entry function '_Z14softmax_kernelIttLi8192ELi4ELi4EEv14Softmax_paramsIT_T0_E' for 'sm_89'
ptxas info    : Function properties for _Z14softmax_kernelIttLi8192ELi4ELi4EEv14Softmax_paramsIT_T0_E
    2624 bytes stack frame, 2680 bytes spill stores, 3416 bytes spill loads
ptxas info    : Used 255 registers, 456 bytes cmem[0]
ptxas info    : Compiling entry function '_Z14softmax_kernelIttLi4096ELi4ELi4EEv14Softmax_paramsIT_T0_E' for 'sm_89'
ptxas info    : Function properties for _Z14softmax_kernelIttLi4096ELi4ELi4EEv14Softmax_paramsIT_T0_E
    544 bytes stack frame, 24 bytes spill stores, 24 bytes spill loads
ptxas info    : Used 255 registers, 456 bytes cmem[0]
ptxas info    : Compiling entry function '_Z14softmax_kernelIttLi2048ELi4ELi4EEv14Softmax_paramsIT_T0_E' for 'sm_89'
ptxas info    : Function properties for _Z14softmax_kernelIttLi2048ELi4ELi4EEv14Softmax_paramsIT_T0_E
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 96 registers, 456 bytes cmem[0]
ptxas info    : Compiling entry function '_Z14softmax_kernelIttLi1024ELi4ELi4EEv14Softmax_paramsIT_T0_E' for 'sm_89'
ptxas info    : Function properties for _Z14softmax_kernelIttLi1024ELi4ELi4EEv14Softmax_paramsIT_T0_E
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 64 registers, 456 bytes cmem[0]
ptxas info    : Compiling entry function '_Z14softmax_kernelIttLi512ELi4ELi4EEv14Softmax_paramsIT_T0_E' for 'sm_89'
ptxas info    : Function properties for _Z14softmax_kernelIttLi512ELi4ELi4EEv14Softmax_paramsIT_T0_E
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 48 registers, 456 bytes cmem[0]
ptxas info    : Compiling entry function '_Z14softmax_kernelIttLi384ELi4ELi2EEv14Softmax_paramsIT_T0_E' for 'sm_89'
ptxas info    : Function properties for _Z14softmax_kernelIttLi384ELi4ELi2EEv14Softmax_paramsIT_T0_E
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 38 registers, 456 bytes cmem[0]
ptxas info    : Compiling entry function '_Z14softmax_kernelIttLi256ELi4ELi4EEv14Softmax_paramsIT_T0_E' for 'sm_89'
ptxas info    : Function properties for _Z14softmax_kernelIttLi256ELi4ELi4EEv14Softmax_paramsIT_T0_E
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 40 registers, 456 bytes cmem[0]
ptxas info    : Compiling entry function '_Z14softmax_kernelIttLi192ELi4ELi2EEv14Softmax_paramsIT_T0_E' for 'sm_89'
ptxas info    : Function properties for _Z14softmax_kernelIttLi192ELi4ELi2EEv14Softmax_paramsIT_T0_E
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 36 registers, 456 bytes cmem[0]
ptxas info    : Compiling entry function '_Z14softmax_kernelIttLi128ELi4ELi4EEv14Softmax_paramsIT_T0_E' for 'sm_89'
ptxas info    : Function properties for _Z14softmax_kernelIttLi128ELi4ELi4EEv14Softmax_paramsIT_T0_E
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 36 registers, 456 bytes cmem[0]
ptxas info    : Compiling entry function '_Z14softmax_kernelIttLi96ELi4ELi1EEv14Softmax_paramsIT_T0_E' for 'sm_89'
ptxas info    : Function properties for _Z14softmax_kernelIttLi96ELi4ELi1EEv14Softmax_paramsIT_T0_E
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 22 registers, 456 bytes cmem[0]
ptxas info    : Compiling entry function '_Z14softmax_kernelIttLi64ELi4ELi2EEv14Softmax_paramsIT_T0_E' for 'sm_89'
ptxas info    : Function properties for _Z14softmax_kernelIttLi64ELi4ELi2EEv14Softmax_paramsIT_T0_E
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 25 registers, 456 bytes cmem[0]
ptxas info    : Compiling entry function '_Z14softmax_kernelIttLi32ELi4ELi1EEv14Softmax_paramsIT_T0_E' for 'sm_89'
ptxas info    : Function properties for _Z14softmax_kernelIttLi32ELi4ELi1EEv14Softmax_paramsIT_T0_E
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 18 registers, 456 bytes cmem[0]
```

### Build convert.cu
```
$ nvcc  -O3 -std=c++17 -ccbin g++ -use_fast_math -Xptxas=-v --expt-relaxed-constexpr -g -lineinfo -DUSE_SAME_SUM_ORDER_IN_SOFTMAX_AS_REF_CODE -DHALF_ACCUMULATION_FOR_FLASH_ATTENTION -I./ -I./generated -I/usr/local/cuda/include -gencode=arch=compute_89,code=\"sm_89\" -c -o convert.cu.o convert.cu
```
```
ptxas info    : 149 bytes gmem
ptxas info    : Compiling entry function '_Z24convert_T_to_fp32_kernelI13__nv_fp8_e5m2EvPvPKvmf' for 'sm_89'
ptxas info    : Function properties for _Z24convert_T_to_fp32_kernelI13__nv_fp8_e5m2EvPvPKvmf
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 16 registers, 380 bytes cmem[0]
ptxas info    : Compiling entry function '_Z24convert_fp32_to_T_kernelI13__nv_fp8_e5m2EvPvPKvmf' for 'sm_89'
ptxas info    : Function properties for _Z24convert_fp32_to_T_kernelI13__nv_fp8_e5m2EvPvPKvmf
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 24 registers, 380 bytes cmem[0]
ptxas info    : Compiling entry function '_Z24convert_T_to_fp32_kernelI13__nv_fp8_e4m3EvPvPKvmf' for 'sm_89'
ptxas info    : Function properties for _Z24convert_T_to_fp32_kernelI13__nv_fp8_e4m3EvPvPKvmf
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 16 registers, 380 bytes cmem[0]
ptxas info    : Compiling entry function '_Z24convert_fp32_to_T_kernelI13__nv_fp8_e4m3EvPvPKvmf' for 'sm_89'
ptxas info    : Function properties for _Z24convert_fp32_to_T_kernelI13__nv_fp8_e4m3EvPvPKvmf
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 24 registers, 380 bytes cmem[0]
ptxas info    : Compiling entry function '_Z24convert_fp32_to_T_kernelI13__nv_bfloat16EvPvPKvmf' for 'sm_89'
ptxas info    : Function properties for _Z24convert_fp32_to_T_kernelI13__nv_bfloat16EvPvPKvmf
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 14 registers, 380 bytes cmem[0]
ptxas info    : Compiling entry function '_Z24convert_fp32_to_T_kernelItEvPvPKvmf' for 'sm_89'
ptxas info    : Function properties for _Z24convert_fp32_to_T_kernelItEvPvPKvmf
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 14 registers, 380 bytes cmem[0]
ptxas info    : Compiling entry function '_Z28convert_int32_to_int8_kernelPvPKvmf' for 'sm_89'
ptxas info    : Function properties for _Z28convert_int32_to_int8_kernelPvPKvmf
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 16 registers, 380 bytes cmem[0]
```

### Build fmha_exe
```
$ g++  -O3 -std=c++17 -g -DUSE_SAME_SUM_ORDER_IN_SOFTMAX_AS_REF_CODE -DHALF_ACCUMULATION_FOR_FLASH_ATTENTION -o fmha_exe fused_multihead_attention.cpp.o fmha_v2_flash_attention_fp16_128_128_S_qkv_16_sm89.cu.o convert.cu.o softmax_fp16.cu.o -L/usr/local/cuda/lib64 -Wl,-rpath=/usr/local/cuda/lib64 -lcudart -lcublas -lcublasLt
```

## 2. Run
```
$ ./fmha_exe -d 16 -b 4 -h 16 -s 1024 -min-s 1024 -fp16 -runs 10 -warm-up-runs 100  -v 1
```
```
$ ./fmha_exe -d 16 -b 4 -h 16 -s 1024 -min-s 1024 -fp16 -runs 1 -warm-up-runs 0  -v 1
Command.......: ./fmha_exe -d 16 -b 4 -h 16 -s 1024 -min-s 1024 -fp16 -runs 1 -warm-up-runs 0 -v 1
Device........: NVIDIA GeForce RTX 4060
Arch.(sm).....: 89
#.of.SMs......: 24
Batch ........: 4
Heads ........: 16
Dimension ....: 16
Dimension of V ....: 16
Seq length ...: 1024
Warm-up runs .: 0
Runs..........: 1

Scale bmm1 ...: 0.250000
Scale softmax.: 1.000000
Scale bmm2 ...: 0.250000

v1=0 il=0 s_q=1024, s=1024 b=4 h=16/16 d=16/16 dtype=FP16, output_dtype=FP16, flash_attn=true_tiled, warp_spec=false, mask=padding, alibi=false, attn=mha, qkv_layout=pack
ed_qkv, wm=4 wn=1
Init .........: Q
Use 1s .......: false
Address ......: 0x00007fd238ca6010
Range ........: 5
Scale ........: 1.000
Values .......: 2.000 2.000 -2.000 0.000 -2.000 -1.000 -2.000 1.000 ...

Init .........: K
Use 1s .......: false
Address ......: 0x00007fd238ca6050
Range ........: 3
Scale ........: 1.000
Values .......: 0.000 1.000 1.000 0.000 -1.000 1.000 -1.000 -1.000 ...

Init .........: V
Use 1s .......: false
Address ......: 0x00007fd238ca6090
Range ........: 5
Scale ........: 0.125
Values .......: -0.125 0.000 -0.125 -0.125 -0.250 0.250 0.250 0.250 ...

Sequence lengths (first 10 batches): 1024, 1024, 1024, 1024, 4096, 
Running fmha_v2 with params: s=1024, b=4, h=16, d=16, dv=16, sm=89, interleaved=0, force_unroll=0, ignore_b1opt=0, force_fp32_acc=0, warp_specialization=0, use_tma=0, use
_flash_attention=1, enable_attn_logit_softcapping=0, attention_input_layout=0, use_tiled=1

Checking .....: O = V * S
Epsilon.......: 0.01500000
Tested........: 1048576
Failed........: 0
Values........: Min=   -0.073792, Max=    0.054413, Avg=  0.000166, Std=  0.010065
Error.........: Min=    0.000000, Max=    0.009313, Avg=  0.000399, Std=  0.000664
Epsilon.......: 0.015000
Infs..........: 0
Checks........: SUCCESS

Non-fused time: 162.131973 ms
Fused time ...: 411.648010 us
Tensor core ..: 10.43 Tflop/s
Bandwidth ....: 20.38 GB/s
Ratio ........: 393.86x
```

## 3. Debug
### --keep --keep-dir ./temp
    在 temp 目录生成 .cu.cudafe1.stub.c 等文件， cuda-gdb device thread 调试会用到

### cuda-gdb
```
$ cuda-gdb ./fmha_exe
(cuda-gdb) b fmha_v2_flash_attention_fp16_128_128_S_qkv_16_sm89_kernel_nl_tiled
(cuda-gdb) directory ./temp
(cuda-gdb) run -d 16 -b 4 -h 16 -s 1024 -min-s 1024 -fp16 -runs 1 -warm-up-runs 0  -v 1
(cuda-gdb) info cuda devices
(cuda-gdb) cuda select thread 0, block (0,0,0), thread (0,0,0)
```

## 4. Notes
### TMA（Tensor Memory Access）
TMA 是 Hopper、Ada Lovelace 引入的高效的内存访问技术，通过硬件加速和自动数据重组，提升 Q、K、V 矩阵加载和输出存储的效率
```
#include <cuda/tma.h>
// 定义TMA操作
tma::Tensor<float, 2> tensor(global_ptr, pitch, shape);
tma::load(register_block, tensor, coords);

// 传统方式：手动加载矩阵块并转置
float a[16][16];
for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 16; j++) {
        a[i][j] = global_mem[base_idx + i * 16 + j];
    }
}
// TMA方式：自动批量加载并转置
tma::Tensor<float, 2> src_tensor(global_mem, shape, stride);
tma::Tensor<float, 2> dst_tensor(shared_mem, transposed_shape, transposed_stride);
tma::copy(dst_tensor, src_tensor);  // 自动处理布局转换
```
### Warp Specialization
SM90 引入 Warp Specialization，允许将 CTA 内的 Threads 分为不同类型（如加载线程束、计算线程束、存储线程束），各自专注于特定任务：
    • 加载线程束：负责从全局内存预取数据到共享内存。    
    • 计算线程束：使用 Tensor Core 执行矩阵运算。    
    • 存储线程束：将结果从共享内存写回全局内存。    
在 FlashAttention 实现中，Warp Specialization 可带来以下优化：
(1) 内存访问优化    
    • 加载线程束：专门处理 Q、K、V 矩阵从全局内存到共享内存的加载，利用 TMA（Tensor Memory Access）技术实现高效批量加载。    
    • 双重缓冲：加载线程束可提前将下一个 Tile 的数据加载到共享内存，与当前 Tile 的计算完全重叠。    
(2) Tensor Core 利用率提升
计算线程束可专注于 Tensor Core 运算，避免被内存操作打断。SM90 的第四代 Tensor Core 支持更高吞吐量的矩阵乘法（如 FP8 精度下的mma.sync指令），与 Warp Specialization 结合后效率进一步提升。
(3) 减少同步开销
传统实现中，线程块内所有线程需通过__syncthreads()同步内存操作和计算。Warp Specialization 允许不同类型的线程束异步工作，仅在必要时进行轻量级同步。

编程接口    
• 线程束 ID 判断：通过threadIdx.x计算当前线程所属的线程束 ID，分配不同任务。
• Memory Fence：使用__threadfence_block() 等轻量级同步替代__syncthreads()。
• TMA 与 Warp 协作：加载线程束使用 TMA 指令批量加载数据，计算线程束直接使用共享内存中的数据。

### Chunked Attention Size
Chunked Attention Size 是指将输入序列分割为小块时，每个小块的长度。它是平衡注意力计算效率与精度的关键参数，在处理超长序列时尤为重要。在 FlashAttention 代码中，这一概念体现为对 Q、K、V 矩阵的Tiling 化处理和循环遍历

### sliding window attention
sliding window attention + cyclic (rolling buffer) kv cache   

### alibi 
alibi :  Attention(Q,K,V)=softmax((QKT/sqrt(d))+bias)V    bias[i][j]=−α∗(i−j)（i是query位置，j是key位置）

### Granular Tiling   ['ɡrænjələr] 
标准 tiling vs granular tiling    
    标准tiling：将矩阵划分为固定大小的块（如 64×64），所有瓦片大小统一。    
    细粒度tiling：根据数据依赖、内存访问模式和硬件特性，动态调整瓦片大小（如 16×16、32×32），甚至允许瓦片重叠或跨维度拆分

### QKV layout
TRTLLM uses 64 by default in paged kv cache.   
    size_t tokens_per_block = 64;    
```
enum class Attention_input_layout
{
    // QKV are packed into [B, S, 3, H, D] layout.
    PACKED_QKV = 0,
    // Q has contiguous [B, S, H, D] layout, while KV has contiguous [B, 2, H, S, D] layout.
    CONTIGUOUS_Q_KV,
    // Q has contiguous [B, S, H, D] layout, while paged KV layout are blocks of indices with shape
    // of [B, 2, Blocks_per_Seq], and the indice indicates the block distance to the pool ptr in
    // global memory.
    Q_PAGED_KV,
};
```

### SageAttention
SageAttention 专为大型语言模型（LLM）设计。它结合了 FlashAttention 的内存访问优化和硬件感知的算法设计，在保持高精度的同时显著提升计算效率。以下是其核心特点：    
1. 硬件感知优化    
SageAttention 针对英伟达 Hopper（SM90）和 Ada Lovelace 架构进行深度优化，充分利用：    
    • 第四代 Tensor Core：支持 FP8、BF16、FP16 等多种精度，加速矩阵乘法。    
    • Warp Specialization：将线程束分为加载、计算、存储三类，实现内存操作与计算的完全重叠。    
    • Tensor Memory Access (TMA)：硬件级优化内存访问，减少数据搬运延迟。    
2. 多精度训练与推理    
SageAttention 支持混合精度计算，尤其在 FP8 精度下表现出色：    
    • FP8 E4M3 格式：专为深度学习设计的 8 位浮点格式，在保持精度的同时减少内存占用和带宽需求。    
    • 动态精度调整：根据计算复杂度自动选择最优精度，平衡性能与精度。    
3. 细粒度内存管理    
通过精细 Tiling 和双重缓冲技术，SageAttention 优化内存访问模式：    
    • 减少全局内存访问：将注意力计算拆分为小瓦片，利用共享内存缓存中间结果。    
    • 内存访问合并：批量加载和存储数据，提高内存带宽利用率。    
4. 因果掩码高效处理    
在自回归生成任务中，SageAttention 高效处理因果掩码（确保模型只关注过去的 token）：    
    • 滑动窗口优化：只计算窗口内的注意力，减少冗余计算。    
    • 掩码预计算：提前处理掩码模式，避免运行时重复计算。    
5. 与 Transformer 架构无缝集成    
SageAttention 设计为可直接替代标准注意力层，无需修改模型架构：    
    • API 兼容性：与 PyTorch、TensorFlow 等主流框架兼容。    
    • 端到端优化：可与其他英伟达库（如 Transformer Engine）协同工作，实现端到端加速。    
6. 性能对比    
与标准注意力和 FlashAttention 相比，SageAttention 在吞吐量和内存效率上进一步提升：    
    • 吞吐量提升：在 FP8 精度下，比 FlashAttention 快 2-3 倍。    
    • 内存占用减少：降低高达 75% 的内存峰值，支持更大批次和模型。    
7. 应用场景    
SageAttention 特别适用于：    
    • 大规模语言模型训练：如 GPT、Llama 等，加速训练过程。    
    • 实时推理：减少响应时间，支持对话系统等交互式应用。    
    • 内存受限环境：如边缘设备或多用户共享 GPU 场景。    


### Sanitize    消毒

### Interleaved memory
标准 QKV 存储 vs Interleaved 存储    
[Q0, Q1, Q2, ..., Qn]  // Q张量    
[K0, K1, K2, ..., Kn]  // K张量    
[V0, V1, V2, ..., Vn]  // V张量    

[Q0, K0, V0, Q1, K1, V1, ..., Qn, Kn, Vn] // 交错存储    

## 5. fused_multihead_flash_attention_kernel_noloop_tiled.h -> device_flash_attention_nl_tiled()

### LDGSTS load
```
    // Load data from memory.
    template <typename Smem_tile>
    inline __device__ void load(Smem_tile& smem_tile)
    {
        uint32_t preds[LDGS];
#pragma unroll
        for (int ii = 0; ii < LDGS; ++ii)
        {
            preds[ii] = row_ + ii * (int) ROWS_PER_LDG < min((int) ROWS, actual_seqlen_);
            preds[ii] &= col_in_bytes_ < VALID_BYTES_PER_ROW;
        }
        // Prepare the load pointers.
        void const* ptrs[LDGS];
#pragma unroll
        for (int ii = 0; ii < LDGS; ++ii)
        {
            ptrs[ii] = q_ptr_ + (int64_t) ii * ROWS_PER_LDG * params_q_stride_in_bytes_;
        }
        // Trigger LDGSTS or the LDGs.
        // The predicates protect against out-of-bound access in rows and cols
        Ldgsts_helper<USE_LDGSTS>::load(this, smem_tile, ptrs, preds);
    }
```
预测加载范围：根据当前线程负责的行和列范围，判断哪些内存访问是有效的（不越界）。    
准备加载指针：计算每个加载操作的全局内存地址。    
执行条件加载：使用预测值保护内存访问，避免越界，并通过 LDGSTS 指令（或普通加载指令）将数据从全局内存搬运到共享内存。    
```
// Push the LDGDEPBAR instruction after the loads for Q, K 
fmha::ldgdepbar<USE_LDGSTS>();
```
ldgdepbar确保：    
    • 预取的数据已完全加载到共享内存。    
    • 后续计算（如矩阵乘法）不会使用未完成加载的数据。    

### softmax 的分段处理和校正  

• global_max：记录当前所有处理过的块中的最大值（用于数值稳定性）。    
• global_sum：记录 softmax 的分母（所有 exp 值的累加和）。    
• reduce < fmha::Max_ >：    计算当前块的每行最大值。    
• apply_exp_with_mask：对当前块应用exp(x - max)，其中max是全局最大值。    
• reduce < fmha::Sum_ >：计算当前块的每行总和。    
• acc_o_normalizer.update：校正之前块的输出，确保所有块的 softmax 计算基于统一的全局最大值    
• acc_o_normalizer.final_update(acc_o, global_sum);  使用最终的全局总和对输出进行归一化，确保所有块的 softmax 结果正确累加    

### 线程（threadIdx.x）任务    
1. data load    
```
// 线程tidx负责加载Q的部分数据到共享内存
Gmem_tile_q gmem_q(params, 0, binfo, tidx, q_loop * Gmem_tile_q::ROWS);
gmem_q.load(smem_q);  // 线程协作加载，tidx决定加载的具体位置
```
2. GEMM, Register File, wmma    
    QKᵀ和 softmax (QKᵀ) V
```
// 线程加载Q/K的子片段到寄存器，参与矩阵乘法
smem_q.load(frag_q[ki], ki);  // tidx决定加载frag_q的哪个部分
smem_k.load(frag_k[ki], ki);
fmha::gemm(acc_p, frag_q[ki], frag_k[ki]);  // 线程协作完成累加
```
由Mma_tile_p/Mma_tile_o定义粒度，如 16x16x16 的子矩阵    

3. Softmax reduction    
每个线程负责计算部分元素的最大值 / 和，再通过 warp 内的__shfl_sync或共享内存汇总，最终得到全局结果。
```
// 线程协作计算当前块的最大值（global_max）, 之和 (global_sum)
softmax.template reduce<fmha::Max_>(global_max);
softmax.template reduce<fmha::Sum_>(global_sum);
```

4. 结果存储，smem -> global mem
```
// 线程加载共享内存中的结果到寄存器，再写入全局内存
smem_o.load(out);  // tidx决定加载out的具体部分
gmem_o.store(out, ii);  // 写入全局内存
```

### CTA/block 任务

blockIdx.z 对应 batch    
blockIdx.y 对应 head    
blockIdx.x 对应序列的分片，代码中有 q_loop 和 o_part 由 blockIdx.x 计算而来，涉及 Q 序列的分片和输出 O 的分区    
1. blockIdx.z: batch      
每个 CTA 处理一个 batch 内的所有计算
```
int const bidb = blockIdx.z; // 用于计算当前批次的 Q/K/V 的起始地址

// 通过binfo（Single_cta类）定位当前批次的全局内存地址
Single_cta<Kernel_traits::VERSION> const binfo(params, bidb, bidh, 0, tidx);
```
2. blockIdx.y：Head
```
int const bidh = blockIdx.y  //用于定位当前头的 Q/K/V 切片

// Head 相关的掩码和偏移计算
mask.load(..., bidh, ...);
```
Q/K/V 被拆分为num_heads个独立的子矩阵，bidh决定当前 CTA 处理第几个子矩阵    
QKᵀ→Softmax→O    

3. blockIdx.x：Sequence    
对应输入序列的分片索引，每个 CTA 处理序列的一个子片段（避免一次性加载过长序列导致的内存溢出）。
```
int const ctas_per_o_row = (Cta_tile_o::VALID_N + Cta_tile_o::N - 1) / Cta_tile_o::N;  // 每个O行需要的CTA数
int const q_loop = blockIdx.x / ctas_per_o_row;  // Q序列的行分片索引
int const o_part = blockIdx.x % ctas_per_o_row;  // 输出O的列分区索引
```
q_loop：决定当前 CTA 处理 Q 序列的哪一段（行范围），如q_sequence_start = q_loop * Gmem_tile_q::ROWS。    
o_part：决定当前 CTA 处理输出 O 的哪一列分区（列范围），如o_part * Cta_tile_o::N    
处理 Q 序列的一个连续分片（行）和 O 的一个连续分区（列），通过循环覆盖整个序列（kv_loop循环）。   

每个 CTA 作为独立单元，负责完成以下端到端计算：    
数据范围划分：通过q_loop/o_part/bidb/bidh定位当前 CTA 处理的 Q/K/V 子矩阵（全局内存地址）。    
共享内存管理：分配共享内存（smem_）用于 Q/K/V 的分片缓存，通过双缓冲（move_to_next_write_buffer）隐藏加载延迟。    
两次 GEMM 计算：完成 QKᵀ（BMM1）和 Softmax (QKᵀ) V（BMM2）的矩阵乘法，通过kv_loop循环处理长序列的分片。    
Softmax 校正：维护全局最大值（global_max）和总和（global_sum），在分片间校正 Softmax 结果（确保与全序列计算一致）    