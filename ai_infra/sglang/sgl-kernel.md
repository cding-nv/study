##

### 新增 normal_decode_metadata_general 内核
CMakeLists.txt：新增 csrc/attention/decode_metadata.cu 源文件
common_extension.cc / common_extension_rocm.cc：注册 torch op
sgl_kernel_ops.h：C++ 声明
attention.py：Python 包装函数
setup_hip.py / setup_rocm.py：编译源文件列表
该 kernel 处理 decode 阶段的 attention metadata 准备（seq_lens → page_table / cu_seqlens_k），支持 SWA（Sliding Window Attention）。

#### 新增自定义 AllReduce 特性 (custom_all_reduce_hip.cuh)
Input Fence 机制（~50 行新增）：

新增 kInputFenceNone / kInputFenceAllThreads / kInputFenceThread0 三种模式
环境变量 SGLANG_CUSTOM_ALLREDUCE_INPUT_FENCE 控制（all/thread0/none）
在 start_sync kernel 中通过 __threadfence_system() 保证跨 GPU 内存一致性
未注册 buffer 路径自动读取环境变量
算法选择（~35 行新增）：

环境变量 SGLANG_CUSTOM_ALLREDUCE_ALGO 强制选择 1-stage 或 2-stage
将原来的 threshold 魔数改为命名常量 kAllReduceSmallThreshold / kAllReduceLargeThreshold
启动方式变更：

从 hipLaunchKernelGGL 改为 hipExtLaunchKernel（通过 stopEvent 获取 kernel 完成时间）
__hip_atomic_store / __hip_atomic_load 改为标准 __atomic_store_n / __atomic_load_n（移除 __HIP_MEMORY_SCOPE_SYSTEM 参数）
新增 hipEvent_t stopEvent 成员，构造时创建、析构时销毁

### 新增 l2norm 内核
CMakeLists.txt：新增 csrc/elementwise/l2norm_kernel.cu
elementwise.py：Python 接口 l2norm(input, eps) = x / sqrt(sum(x²) + eps)
common_extension.cc / common_extension_rocm.cc：torch op 注册
init.py：公开导出
setup_hip.py / setup_rocm.py：编译列表

### 新增 HCU KV Cache Transfer 内核（transfer.cu 最大改动）
新增 _hcu 后缀的 KV cache 传输函数簇，支持 HCU 平台的不同内存布局和传输路径：
```
transfer_kernel_impl_hcu — 新增 _hcu 版 transfer kernel，支持 per-layer 批量处理
transfer_item_warp_hcu — 使用 __builtin_nontemporal_load/store 的 warp 级数据搬运
get_global_offset_lf_tbl_hcu / pf_hcu / lf_hcu — HCU 的内存布局偏移计算
get_rocm_kernel_accessible_ptr — 通过 cudaHostGetDevicePointer 获取主机内存的设备可访问指针（host tensor → device pointer mapping）
transfer_kv_launcher_hcu — HCU 专用 launcher
4 个顶层 Python 包装：transfer_kv_all_kernel_lf_pf_D2H_hcu / transfer_kv_per_layer_kernel_pf_lf_H2D_hcu / transfer_kv_all_direct_pf_lf_H2D_hcu / transfer_kv_all_direct_lf_pf_D2H_hcu
```
1. CU KV Cache Transfer 的核心用途是 HiCache — KV Cache 的 GPU ↔ CPU 数据传输（Swap In / Swap Out）
在 LLM 推理服务中，GPU 显存有限，当并发请求过多、KV cache 超出显存容量时，需要将部分请求的 KV cache 换出到 CPU 内存，等需要时再换回 GPU。这就是这组 transfer_kv_*_hcu 函数做的事情，调用方在 memory_pool_host.py 中：
|Python 调用方	|方向	|含义|
|load_to_device_per_layer()	|H2D (Host→Device)	|KV cache 从 CPU 加载回 GPU（Swap In）|
|backup_from_device_all_layer()	|D2H (Device→Host)	|KV cache 从 GPU 卸载到 CPU（Swap Out）|

2. 两种 IO 后端
```
if io_backend == "kernel":   # GPU kernel 路径 — 高性能
    transfer_kv_per_layer_kernel_pf_lf_H2D_hcu(...)    # H2D
    transfer_kv_all_kernel_lf_pf_D2H_hcu(...)          # D2H
elif io_backend == "direct": # copy_() 路径 — 简单直接
    transfer_kv_all_direct_pf_lf_H2D_hcu(...)          # H2D
    transfer_kv_all_direct_lf_pf_D2H_hcu(...)          # D2H
```    
* kernel 路径：启动 GPU kernel，用 __builtin_nontemporal_load/store（non-temporal = 绕过 cache，适合大数据块搬运）逐 warp 搬运
* direct 路径：直接用 torch.Tensor.copy_() 做 host↔device 拷贝

3. 布局转换：Page-First ↔ Layer-First
函数名中的 pf / lf 是关键——两侧内存布局不同，传输时要做隐式转置：

布局	|存储格式	|在哪一侧
|---|---|---|
pf (Page-First)	|[page_id][layer_id][token_bytes]	|GPU 侧（paged attention 的标准布局）
lf (Layer-First)	|[layer_id][page_id][token_bytes]	|CPU 侧（按 layer 组织，便于管理和分配）
对应三个 offset 计算函数：
```
get_global_offset_pf_hcu → base + page_id * page_dim + layer_id * item_size_bytes
get_global_offset_lf_hcu → base + layer_id * layer_dim + page_id * item_size_bytes
get_global_offset_lf_tbl_hcu → layer_base_tbl[layer_id] + page_id * item_size_bytes（layer 之间不连续，用指针表索引）
```
4. transfer_item_warp_hcu 的作用（你选中的函数）    
这是 GPU kernel 中最底层的搬运单元。每个 warp lane 以 uint64_t 为单位，用 non-temporal 指令搬运一个 KV token 的字节数据：

```
// 读: 绕过 L1/L2 cache（因为这是一次性的大块数据，cache 没意义）
uint64_t tmp = __builtin_nontemporal_load(src + j);
// 写: 同样绕过 cache
__builtin_nontemporal_store(tmp, dst + j);
```
对比 NVIDIA 路径用的是 PTX 内联汇编 ld.global.nc / st.global.cg，语义相同。

5. 为什么需要 get_rocm_kernel_accessible_ptr    
这是 HCU 平台特有的处理：CPU 侧的 tensor 可能是 host-pinned memory（通过 pin_memory=True 分配），GPU kernel 不能直接用 host 指针访问，必须通过 cudaHostGetDevicePointer 获取对应的 device-visible 指针。这就是之前在 transfer_kv_launcher 和 transfer_kv_launcher_hcu 中 #ifdef USE_ROCM 分支做的事情：  

```
src_k_ptr = get_rocm_kernel_accessible_ptr(src_k);  // host ptr → device-visible ptr
```
一句话总结    
这组 transfer_kv_*_hcu 函数是 HiCache 在 HCU 平台上的 GPU↔CPU KV cache 数据传输引擎，负责在内存压力下将 KV cache 换出到 CPU 或换回 GPU，同时做 page-first ↔ layer-first 的布局转换。

6.transfer_item_warp_hcu 是否同时支持 H2D 和 D2H？    
  是的，完全对称。 它只是对 src_addr 做 non-temporal load、对 dst_addr 做 non-temporal store，不关心方向：
```
// 第 94-95 行 — 纯数据搬运，src/dst 只是指针
uint64_t tmp = __builtin_nontemporal_load(src + j);   // 读
__builtin_nontemporal_store(tmp, dst + j);            // 写
```
方向由调用的 wrapper 函数决定——看传进来的 src/dst 分别指向 host 还是 device：

7. Device 如何访问 Host 内存？    
这分两个层面：    
层面一：Host 内存必须先 Pin（锁页）    
在 memory_pool_host.py:144 中，host buffer 分配后会通过 cudaHostRegister 注册为 pinned memory：    
```
if pin_memory:
    torch.cuda.cudart().cudaHostRegister(
        buffer.data_ptr(), buffer.numel() * buffer.element_size(), 0
    )
```
锁页后的内存在物理上不会被 OS 换出，GPU 可以通过 PCIe 总线直接访问它。    

层面二：Kernel 使用前做指针转换    
Pinned memory 的 CPU 指针不能直接传给 GPU kernel——GPU 需要设备可见的地址。这在 kernel launch 之前通过 get_rocm_kernel_accessible_ptr() 完成，位于 transfer.cu:38-56：
```
void* get_rocm_kernel_accessible_ptr(const at::Tensor& tensor) {
    void* ptr = tensor.data_ptr();
    if (tensor.is_cuda() || ptr == nullptr) {
        return ptr;   // 已经是 GPU tensor，直接返回
    }
    // CPU tensor → 获取设备可见指针
    void* device_ptr = nullptr;
    cudaError_t err = cudaHostGetDevicePointer(&device_ptr, ptr, 0);
    return device_ptr;  // 返回 GPU 可直接访问的地址
}
```
调用时机（在 transfer_kv_launcher_hcu:532-537 中）：
```
// 对 4 个 buffer 指针全部做转换
src_k_ptr = get_rocm_kernel_accessible_ptr(src_k);  // 如果 src_k 在 host → 转换为 device-visible ptr
dst_k_ptr = get_rocm_kernel_accessible_ptr(dst_k);  // 如果 dst_k 在 host → 同样转换
src_v_ptr = get_rocm_kernel_accessible_ptr(src_v);
dst_v_ptr = get_rocm_kernel_accessible_ptr(dst_v);
// 转换后的指针传给 kernel，GPU 就可以直接读写了
transfer_kernel_impl_hcu<<<grid, block, 0, stream>>>(src_k_ptr, dst_k_ptr, ...);
````
关键点：不需要显式 cudaMemcpy——GPU kernel 启动后，通过 cudaHostGetDevicePointer 返回的地址，直接通过 PCIe 对 host pinned memory 做 load/store（non-temporal 指令写的是 PCIe BAR 地址空间）。这是一种 zero-copy 方式，省去了额外的拷贝开销。

8. grid_dim， threads_per_block  如何确定？   
在 transfer.cu:521-525：
```
const int64_t num_items = src_indices.numel();       // 例如 128 条记录
const int64_t num_pages = num_items / page_size;     // page_size=16 → 8 个 page
dim3 grid_dim(num_pages, 1, 1);                     // 每个 page 一个 block
const int32_t threads_per_block = num_warps_per_block * WARP_SIZE;
//       ^ 例如 16 warps × 64 (WARP_SIZE) = 1024 threads
```
核心理念：一个 block 负责一个 page，block 内所有线程协作搬运该 page 的 K+V 数据。
```
// kernel 内部 — 各 block 独立处理自己的 page
int32_t page_index_id = blockIdx.x;                         // block 0 → page 0, block 1 → page 1 ...
const int64_t s_page_id = src_indices[page_index_id * page_size] / page_size;
const int64_t d_page_id = dst_indices[page_index_id * page_size] / page_size;

for (each layer) {
    src_ptr = offset(s_page_id, layer_id);                  // 计算源地址
    dst_ptr = offset(d_page_id, layer_id);                  // 计算目标地址
    transfer_item_warp_hcu(lane_id, src_ptr, dst_ptr, ...); // 全 block 线程一起搬
}
```
原始版本 (transfer_kv_launcher) 对比
```
// 原始：更细粒度，每个 index 条目就是一个 page_id
const int64_t num_items = src_indices.numel();              // 128 个 page
const int64_t items_per_warp = div_up(num_items, block_quota * num_warps_per_block);
const int32_t num_blocks = div_up(num_items, items_per_warp * num_warps_per_block);
// grid: ceil(128 / (items_per_warp * num_warps_per_block)) 个 block
// 每个 warp 处理 items_per_warp 个 page，用 for 循环串行
```
维度	|原始版本	|HCU 版本
|---|---|---|
grid 维度	|num_blocks（根据 items/warp 计算）	|num_pages（1 block / page）
索引粒度	|src_indices[i] = 1 个 page_id	|page_size 个连续条目 = 1 个 page
warp 调度	|每个 warp 处理多个 page	|整个 block 处理 1 个 page（所有 layer）
适用场景	|通用 page table	|HCU 的 layout_hcu 特殊 index 格式

9.为什么不用 cudaMemcpy 类似的方式？      
有 四个 原因，任何一个都足以排除 cudaMemcpy：

a. 布局不同 — cudaMemcpy 做不到    
source 和 destination 的内存布局不同，需要在拷贝过程中做地址重映射：
```
GPU (Page-First):                    CPU (Layer-First):
page 0                                layer 0
├─ layer 0: token [0..N]             ├─ page 0: token [0..N]
├─ layer 1: token [0..N]             ├─ page 1: token [0..N]
├─ layer 2: token [0..N]             ├─ page 2: token [0..N]
└─ ...                               ├─ ...
page 1                                layer 1
├─ layer 0: token [0..N]             ├─ page 0: token [0..N]
├─ layer 1: token [0..N]             ├─ page 1: token [0..N]
├─ layer 2: token [0..N]             ├─ ...
└─ ...
                                     └─ ...
```
cudaMemcpy 只能做连续字节块的线性拷贝，它能从 src + offset 拷贝 N 字节到 dst + offset，但无法做到：
* 从 [page 0, layer 0] → [layer 0, page 0] 这种跨步长 scatter/gather
* 每层、每页的地址都是通过 SrcOffsetFn / DstOffsetFn 动态计算的

b. 稀疏 / 索引访问    
只传输 指定的 page，不是整个 buffer。src_indices / dst_indices 指定了哪些 page 需要搬：
```
const int64_t s_page_id = src_indices[page_index_id * page_size] / page_size;
const int64_t d_page_id = dst_indices[page_index_id * page_size] / page_size;
```
cudaMemcpy 要求源和目标都是连续的地址范围，无法按索引列表做批量稀疏拷贝。cudaMemcpy2D 或 cudaMemcpy3D 能处理步长，但步长必须是常量，不支持 per-page 变化的索引。

c. Non-temporal 访问 — 避免污染 L2 cache    
KV cache 数据体积大、一次性使用（换出后短期内不会再访问），放进 cache 毫无意义反而会驱逐有用的数据。所以用：
```
__builtin_nontemporal_load(src + j);   // 读: 绕过 cache
__builtin_nontemporal_store(tmp, dst + j);  // 写: 也绕过 cache
cudaMemcpy 走的是 DMA 引擎，其 cache 行为由硬件决定，开发者无法控制。
```    

d. 单次 kernel 完成多层搬运，减少 launch 开销    
一个 kernel launch 同时传输所有 layer 的 K+V：
```
for (int64_t layer_id = start_layer_id; layer_id < start_layer_id + num_layers_to_process; ++layer_id) {
    // 搬运 K
    transfer_item_warp_hcu(lane_id, src_ptr_k, dst_ptr_k, ...);
    // 搬运 V
    transfer_item_warp_hcu(lane_id, src_ptr_v, dst_ptr_v, ...);
}
```
如果用 cudaMemcpy，每个 layer 的 K 和 V 需要各一次调用，N 层模型就是 2N 次 API 调用。    
总结:  cudaMemcpy 做的是"搬一块连续内存"；而这里需要的是 "按索引表做跨布局的数据搬移 + 隐式转置"。本质上是 GPU 上的 scatter/gather + transpose 操作，必须用自定义 kernel。
