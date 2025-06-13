#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

__global__ void findFirstIndices(int* in, int n, int* out) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > n) return;

    if (tid == 0 || in[tid] != in[tid - 1]) {
        out[tid] = tid;
    } else {
        out[tid] = -1;
    }
}

int main() {
    // 示例输入（已排序，可包含重复）
    std::vector<int> h_in = {0, 1, 1, 3, 3, 3, 4, 5, 5, 5, 6, 7};

    int n = static_cast<int>(h_in.size());
    std::vector<int> h_out(n);

    // 分配 GPU 内存
    int *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in, n * sizeof(int));
    cudaMalloc(&d_out, n * sizeof(int));

    // 拷贝输入到设备
    cudaMemcpy(d_in, h_in.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    // 启动 kernel
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    findFirstIndices<<<gridSize, blockSize>>>(d_in, n, d_out);

    // 拷回结果
    cudaMemcpy(h_out.data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost);

    // 释放 GPU 内存
    cudaFree(d_in);
    cudaFree(d_out);

    // 验证并打印每个值的首次出现下标
    printf("值 -> 第一次出现下标\n");
    for (int i = 0; i < n; ++i) {
        if (h_out[i] != -1) {
            printf("%d -> %d\n", h_in[i], h_out[i]);
        }
    }

    return 0;
}

