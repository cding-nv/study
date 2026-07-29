
## python/sglang/jit_kernel

`jit_kernel` 是 SGLang 的**运行时 GPU kernel JIT 编译系统**，与 `sgl-kernel` 构成 SGLang kernel 的两层架构：

| | sgl-kernel (AOT) | jit_kernel (JIT) |
|---|---|---|
| 编译时机 | 构建时 (CMake) | 运行时首次调用 |
| 产物 | 预编译 `.so`/`.dll` | 临时 `.so`（进程内缓存） |
| 模板参数 | 固定（少数重载） | 运行时确定（大量组合） |
| 分发形式 | 打包在 wheel 中 | 源码头文件 (`.cuh`) |
| 典型 kernel | allreduce, attention, norm | DeepSeek V4 compress, hadamard, topk |
| Kernel 源 | `.cu` / `.hip` | `.cuh` (header-only 模板) |
| 运行时依赖 | CUDA/HIP Runtime | CUDA/HIP Toolkit (完整编译器) |

### 为什么需要 JIT kernel？

GPU kernel 通常需要针对**具体参数组合**做模板特化以获得最佳性能。以 DeepSeek V4 的 compress kernel 为例：

```cpp
template <typename DType, int64_t kHeadDim, int64_t kCompressRatio, bool kUsePDL>
__global__ void flash_c128_decode(const Compress128DecodeParams params) { ... }
```
参数组合空间：

|参数	|可能值	|数量|
|---|---|---|
|DType|fp16, bf16, fp8	|~3|
|kHeadDim|128, 192, 256, 512, ...	|~6|
|kCompressRatio|4, 128	|2|
|kUsePDL|true, false	|2|
组合总数 ≈ 3 × 6 × 2 × 2 = 72 个变体。如果全部 AOT 编译，二进制体积急剧膨胀，而且大部分变体用户根本用不到。
JIT 的做法是：运行时根据模型的实际 dtype 和 head_dim，只编译需要的那一个变体。

### Python 调用模式
```
# hadamard.py — 典型的 jit_kernel 用法
@cache_once                                    # 同一进程内只编译一次
def _jit_hadamard_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)                # Python dtype → C++ 模板参数
    return load_jit(                            # 调用 hipcc/nvcc 编译 + dlopen
        "hadamard", *args,
        cuda_files=[".../hadamard_jit.cuh"],    # .cuh 头文件（不是 .cu）
        cuda_wrappers=[
            ("hadamard_transform", f"HadamardKernel<{args}>::run"),
        ],
    )
```
调用链：

```
Python: hadamard_transform(tensor, scale)
  → _jit_hadamard_module(tensor.dtype)       # @cache_once 缓存
    → load_jit("hadamard", dtype=bf16, ...)  # 首次: 调 hipcc 编译 .cuh → .so
      → ctypes/dlopen 加载 .so
  → module.hadamard_transform(x, out, scale)  # tvm_ffi 桥接到 C++ 代码
```
### 为什么不放在更底层（sgl-kernel C++）？
根本原因是编译时机的差异，有三个硬约束：
1. AOT 无法覆盖参数组合爆炸    
sgl-kernel 用 CMake 预编译，必须枚举所有变体。jit_kernel 的模板参数值由模型文件决定（head_dim 在模型加载时才知道，dtype 取决于用户配置），构建时根本不知道要编译哪些。

2. 编译工具链必须在运行时可用    
JIT 编译需要 nvcc / hipcc 在部署机器上存在。sgl-kernel 的 wheel 分发到没有 CUDA Toolkit 的机器也能运行（只需 CUDA Runtime），但 jit_kernel 要求完整的编译工具链。这是两种不同的部署假设。

3. 依赖 tvm_ffi Python↔C++ 桥接层    
jit_kernel 依赖 tvm_ffi 做 JIT 编译和动态加载。tvm_ffi 是 Python 库，提供：

   * load() / load_inline() — 调用编译器 + 生成 .so + dlopen
   * TensorView — Python tensor 到 C++ kernel 参数的类型安全映射
   * TensorMatcher — 运行时 shape/dtype/device 校验
整个工作流是 Python 驱动的（Python 决定何时编译、用什么参数），放在 C++ 层反而不自然。