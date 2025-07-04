对拆法
 0 0 0 0 | 0 0 0 0
```
for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride)
        shared[threadIdx.x] += shared[threadIdx.x + stride];
    __syncthreads();
}
```
stride:  4, 2, 1   stride 和 每次活跃的 thread 个数相等
sdata[tid] += sdata[tid + stride]
for stride >>= 1
  if tid < stride 

相邻法
```
__global__ void sum(int* input) {
    const int tid = threadIdx.x;

    auto step_size = 1;
    int number_of_threads = blockDim.x;

    while (number_of_threads > 0) {
        if (tid < number_of_threads) {// still alive?
            const auto fst = tid * step_size * 2;
            const auto snd = fst + step_size;
            input[fst] += input[snd];
        }

        step_size <<= 1;
        number_of_threads >>= 1;
    }
}
```
for (int stride = 1; stride )



block 间通信解决，
    cpu 累加
    global memory
    cooperative group 跨 block 通信
        kernel 由 cudaLaunchCooperativeKernel 启动；
        线程数 < 设备支持的最大 cooperative launch 限制。
        支持 grid-wide 的同步和通信；
        可以在一个 kernel 中实现 block 间的 “显式同步”；

shuffle 指令
#define N 1024000
#define THREADS_PER_BLOCK 256

// warp 内的 sum，使用 shuffle 指令
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// 每个 block 负责一段，结果写入 output[blockIdx.x]
__global__ void simple_reduce_kernel(float* input, float* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = 0.0f;

    if (idx < N) val = input[idx];

    // warp 内做 reduction
    val = warpReduceSum(val);

    // 每个 warp 的 lane 0 写入结果（假设 blockDim.x 是 warpSize 的倍数）
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(output, val);
    }
}


## & 31,  即 每个 warp 累加的结果 val 都会 atomicAdd 到 output