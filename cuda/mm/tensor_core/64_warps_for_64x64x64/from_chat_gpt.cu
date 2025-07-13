#include <mma.h>
#include <cuda_fp16.h>
#include <stdio.h>

using namespace nvcuda;

#define M 64
#define N 64
#define K 64

//  Tensor Core 编程采用 warp-scope fragment
// 一个 warp 跑 4 次 完成 C 的 一个 tile 16x16, 每次 16x16x16
// 最后 每个 thread 都持有这个 16x16=256 元素中的 8 个
// fragment<accumulator, 16, 16, 16, float>, fragment.num_elements == 8

// Kernel：使用 WMMA 完成 64x64x64 GEMM
__global__ void wmmaGEMM64x64x64(const half *A, const half *B, float *C) {
    // blockIdx.x, blockIdx.y 对应 tile 在 C 中的位置
    int tile_row = blockIdx.y;
    int tile_col = blockIdx.x;

    // 每个 block 一个 warp（32 threads）
    // 每个 block 计算一个 16x16 的 tile，tile size 固定为 16x16x16
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    // 初始化输出为 0
    wmma::fill_fragment(c_frag, 0.0f);

    // 遍历 K 维度的 tile
    for (int tile_k = 0; tile_k < K; tile_k += 16) {
        const half* tile_ptr_a = A + (tile_row * 16) * K + tile_k;
        const half* tile_ptr_b = B + tile_k * N + tile_col * 16;

        // 注意：A 是 row_major，B 是 col_major
        wmma::load_matrix_sync(a_frag, tile_ptr_a, K);
        wmma::load_matrix_sync(b_frag, tile_ptr_b, N);

        // Tensor Core 执行 GEMM
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // 写回 C
    float* tile_ptr_c = C + tile_row * 16 * N + tile_col * 16;
    wmma::store_matrix_sync(tile_ptr_c, c_frag, N, wmma::mem_row_major);
}

// 初始化数据
void init_host_matrix(half *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = __float2half(static_cast<float>(i % 3 + 1));  // 简单数据
    }
}

int main() {
    // host 内存
    half *h_A = new half[M * K];
    half *h_B = new half[K * N];
    float *h_C = new float[M * N];

    init_host_matrix(h_A, M, K);
    init_host_matrix(h_B, K, N);

    // device 内存
    half *d_A, *d_B;
    float *d_C;
    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, K * N * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, M * N * sizeof(float));

    // 每个 16x16 tile 对应一个 block，共需 4x4 个 tile
    dim3 gridDim(4, 4);
    dim3 blockDim(32);  // 每个 warp 32 threads

    wmmaGEMM64x64x64<<<gridDim, blockDim>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印部分结果
    printf("Result C[0..3][0..3]:\n");
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j)
            printf("%8.1f ", h_C[i * N + j]);
        printf("\n");
    }

    // 清理
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

