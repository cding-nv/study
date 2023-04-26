#include <cuda_runtime.h>
#include <iostream>

#include <numeric>
using namespace std;

// Both source and destination threads in the warp must “participate”
// Sync “mask” used to identify and reconverge needed threads
__global__ void reduce_ws(float *gdata, float *out, int size) {
    __shared__ float sdata[32];
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    float val = 0.0f;
    unsigned mask = 0xFFFFFFFFU;
    int lane = threadIdx.x % warpSize;
    int warpID = threadIdx.x / warpSize;
    while (idx < size) { // grid stride loop to load
        val += gdata[idx];  // 在 warp 中的每个 thread 负责 load 一个对应 id 的 data 到 register 中
        idx += gridDim.x * blockDim.x; // 每个 thread 对应 id data 累加， 这部分没有用 shuffle 指令，看来不是最优的，
                                       // 但都是累加到寄存器，可能有 register 访问 bank conflict 的问题， arbiter序列化访问和operand collector 缓存
                                       // 用来解决这个问题。Load/store 指令可以 gather/scatter 一次性读写大块数据.
    }

    // 1st warp-shuffle reduction
    for (int offset = warpSize/2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(mask, val, offset); // 一个 warp 包含 32 个 lanes， 每个 thread 占用一个 lane
                                                    // warpSize 是 32， 循环 5 次， offset 分别为 16， 8， 4， 2， 1
                                                    // 最终 Warp 中的第一个 thread 的 val 是累加结果，
                                                    // 最后一个循环只有 第一个thread 是 active 的
    if (lane == 0) sdata[warpID] = val;  // 数据分成了 32 份， 每份用一个 warp 完成累加
    __syncthreads(); // 等每个 warp 完成任务 并写入 shared mem 中
    // 最后 32 份数据的累加让 warp 0 来干
    if (warpID == 0){
        // reload val from shared mem if warp existed
        val = (tid < blockDim.x / warpSize) ? sdata[lane] : 0; // 将share mem 里的数据 load 到 register 中
        // final warp-shuffle reduction
        for (int offset = warpSize/2; offset > 0; offset >>= 1)
            val += __shfl_down_sync(mask, val, offset);  // 将 32 份数据累加
        if (tid == 0) atomicAdd(out, val);  // 将 register 数据写到 global address
    }
}

int main() {
    int N = 1024000;
    float* g_idata;
    float* g_odata;
    cudaMallocManaged(&g_idata, N * sizeof(float));

    int sum = 0;
    for(int i = 0; i < N; i++) {
      int rand_num = rand() % 10;
      g_idata[i] = rand_num;
      sum += rand_num;
    }
    std::cout << "CPU sum: " << sum << std::endl;

    int block_size = 1024;
    int num_blocks = (N + block_size -1) / block_size;

    cudaMallocManaged(&g_odata, sizeof(int));
    reduce_ws<<<num_blocks, block_size>>>(g_idata, g_odata, N);
    cudaDeviceSynchronize();

    std::cout<<"GPU sum: "<< g_odata[0] <<std::endl;
}
