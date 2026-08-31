---
name: DCU-kernel-optimize
description: 系统性 HIP kernel 调优指南，面向海光 DCU 架构（BW151/BW1101）。涵盖合并内存访问与对齐、向量化加载（float4）、Bank Conflict 避免、异步拷贝与 Stream 重叠、Occupancy 与波前调度（launch_bounds/寄存器压力）、LDS 共享内存使用与 Tree 规约、向量化与内置函数（fp16/bf16/fast_math）、Wavefront Shuffle、分支发散避免、寄存器数组下标常量化、VGPR 节省（SGPR 利用）、GEMM Tiling 案例。附带性能瓶颈速查表。当用户要求优化 HIP kernel、提升 DCU 性能时自动触发。
argument-hint: "[HIP kernel 文件路径或代码片段]"
allowed-tools:
  - Read
  - Edit
  - Write
  - Glob
  - Grep
  - Bash
---
# HIP Kernel 调优技能

本技能指导你如何系统性地分析和优化 HIP (hygon DCU编程) 内核，提升在 hygon DCU 上的计算性能。

## 适用场景
- 已有可运行的 HIP 内核，但性能不达标
- 需要针对特定 hygon DCU 架构（如 BW151/BW1101系列）进行优化
- 希望了解 ROCm 工具链的性能分析流程

## 调优步骤概览
1. 内存访问优化
2. 计算与访存重叠
3. 占用率与波前调度
4. 使用 LDS（共享内存）
5. 向量化与内置函数
6. 微架构特性利用
7. 迭代验证

## 详细调优技术

### 1. 内存访问优化

**合并访问（Coalescing）**：
HIP 中线程按wave（64 线程）执行，需确保相邻线程访问连续内存：

```cpp
// 坏：跨步访问，每个波前产生多次内存事务
__global__ void bad_coalesce(float* in, float* out) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    out[tid] = in[tid * 32];  // stride = 32，严重非合并
}

// 好：连续访问
__global__ void good_coalesce(float* in, float* out) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    out[tid] = in[tid];  // 连续，完美合并
}
```

**对齐与宽加载**：
- 确保数据地址对齐到 128 字节（`__align__(16)` for float4）
- 使用 `float4`/`double2` 等向量类型提高带宽利用率

```cpp
// 使用 float4 一次加载 16 字节
__global__ void vector_load(float4* in, float4* out) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float4 data = in[tid];
    data.x += 1.0f;  // 处理
    out[tid] = data;
}
```

**避免 bank conflict**：
LDS 有 32 个 bank（每个 4 字节），多个线程访问同一 bank 会串行化。

```cpp
// 冲突模式
__shared__ float smem[256];
int tid = threadIdx.x;
float val = smem[tid];           // 无冲突（每个线程不同 bank）
float val2 = smem[tid * 2];      // 可能冲突（若步长倍数导致 bank 碰撞）

// 缓解：填充或改变索引模式
__shared__ float smem[256 + 32]; // 添加填充避免 bank 冲突
```

### 2. 计算与访存重叠

**使用异步内存拷贝**：
利用 `hipMemcpyAsync` 与计算流重叠：

```cpp
hipStream_t stream1, stream2;
hipStreamCreate(&stream1);
hipStreamCreate(&stream2);

// 流 1：数据传输与内核执行重叠
hipMemcpyAsync(d_a, h_a, size, hipMemcpyHostToDevice, stream1);
hipLaunchKernelGGL(kernel, grid, block, 0, stream1, d_a, d_b);
hipMemcpyAsync(h_b, d_b, size, hipMemcpyDeviceToHost, stream1);

// 流 2：独立计算
hipLaunchKernelGGL(another_kernel, grid, block, 0, stream2, d_c, d_d);
```



### 3. 占用率与波前调度

**计算最佳线程块大小**：
每个计算单元 (CU) 有 4 个 SIMD，每个 SIMD 最多 10 个wave。

```cpp
// 查询设备属性
hipDeviceProp_t props;
hipGetDeviceProperties(&props, 0);
int maxThreadsPerCU = props.warpSize * props.maxThreadsPerMultiProcessor; 
// 通常建议 blockDim.x 为 64, 128, 256 等 64 的倍数
```

**资源限制检查**：
- 寄存器使用：`__launch_bounds__` 指导编译器优化
```cpp
__global__ void __launch_bounds__(256, 1)  // 最大线程 256
my_kernel(float* in) { ... }
```
- 共享内存：`extern __shared__ char smem[];` 动态分配

**减少寄存器压力**：
```cpp
// 使用 volatile 或显式溢出
#pragma unroll
for(int i=0; i<4; i++) {
    float tmp = ...;  // 编译器可能将其溢出到 LDS
}
// 更好的：手动使用 LDS 或限制变量作用域
```

### 4. 高效使用 LDS（共享内存）

**LDS 延迟约 40-60 cycle，比全局内存快得多**。

**规约操作示例**：
```cpp
__global__ void reduce_lds(float* in, float* out) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    smem[tid] = in[idx];
    __syncthreads();
    
    // 树形规约
    for(int s = blockDim.x/2; s > 0; s >>= 1) {
        if(tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    
    if(tid == 0) out[blockIdx.x] = smem[0];
}
```

**LDS 配置**：
```cpp
// 动态 LDS
size_t lds_size = blockDim.x * sizeof(float);
hipLaunchKernelGGL(kernel, grid, block, lds_size, stream, ...);
```

### 5. 向量化与内置函数

**使用 hip::builtins**：
```cpp
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>

// 使用 half 精度提高吞吐量
__global__ void half_kernel(half* in, half* out) {
    half val = in[threadIdx.x];
    out[threadIdx.x] = __hadd(val, __float2half(1.0f));
}
```

**快速数学函数**：
```cpp
// 编译选项: --fast-math
float r = sqrtf(x);      // 使用快速近似 sqrt
float s = __cosf(x);     // 显式使用内建函数
```

**使用向量加载存储指令**：
```cpp
// 自动向量化由编译器完成，但可手动提示
#pragma clang loop vectorize(enable)
for(int i=0; i<N; i++) {
    out[i] = in[i] * 2.0f;
}
```

### 6. 微架构特性

**利用 Wavefront 内建函数**：
```cpp
// 获取波前内索引
int lane_id = threadIdx.x & 63;
int wave_id = threadIdx.x >> 6;

// 波前内洗牌指令 (类似 CUDA __shfl)
float val = __shfl_up(val, 1);  // 从上一 lane 取值
```

**避免分支发散**：

```cpp
// 坏：波前内分支
if(threadIdx.x % 2 == 0) {
    // 只有一半线程执行，另一半空闲
}

// 好：使用掩码或重构为两个循环
if(threadIdx.x < 32) {
    // 整个波前执行相同路径
}
```

### 7.寄存器数组下标要用常数

1.寄存器数组下标要用常量，或者可展开的循环

```
     const int Loop =10;
     for(int i=0; i<Loop; i++)   //Loop是常量，循环可展开
    {                                      
        c[i] = a[i]+ b[i]              
    }
    
     const int Loop = len;
     for(int i=0; i<Loop; i++)   //Loop是变量，循环不可展开
    {                                      
        c[i] = a[i]+ b[i]              
    }
```

### 8.节省VGPR使用量

节省vgpr，非threadId相关的变量可以用sgpr，如warpId

  int warp_id_vec = threadIdx.x / 64; //warp_id_vec 是vgpr

  warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec);  //warp_id是sgpr

```bash

```

## 常见性能瓶颈速查表

| 现象         | 可能原因              | 解决方案                                 |
| ------------ | --------------------- | ---------------------------------------- |
| 占用率低     | 寄存器过多 / LDS 过大 | 减少变量作用域，使用 `__launch_bounds__` |
| 内存带宽低   | 非合并访问            | 调整线程块布局，使用向量类型             |
| 计算单元闲置 | 分支发散 / 同步开销大 | 重构分支，减少 `__syncthreads()`         |
| L2 命中率低  | 数据重用差            | 使用 LDS 缓存，调整 tile 大小            |
| 峰值计算未达 | 未使用向量指令        | 手动向量化，启用 fast-math               |

## 典型调优案例

**矩阵乘法 Tiling**：
```cpp
// 使用 LDS 分块，每个块 16x16
__global__ void matmul(float* A, float* B, float* C, int N) {
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    float sum = 0.0f;
    for(int k = 0; k < N/16; k++) {
        As[ty][tx] = A[(by*16 + ty)*N + (k*16 + tx)];
        Bs[ty][tx] = B[(k*16 + ty)*N + (bx*16 + tx)];
        __syncthreads();
        
        for(int i = 0; i < 16; i++)
            sum += As[ty][i] * Bs[i][tx];
        __syncthreads();
    }
    C[(by*16 + ty)*N + (bx*16 + tx)] = sum;
}
```
调整块大小 (16->32) 并使用向量加载可进一步提升。

## 总结

遵循以下顺序进行调优：
1. 确保功能正确，获取基线
2. 用 xprof识别瓶颈
3. 针对瓶颈应用上述优化技术
4. 每次只改一个变量，测量效果
5. 迭代直至性能稳定

记住：**过度优化是万恶之源**。达到目标性能即可停止。