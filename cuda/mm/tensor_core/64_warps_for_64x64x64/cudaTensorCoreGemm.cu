#include <assert.h>
#include <stdio.h>
#include <cuda.h>
#include <mma.h>
#include <helper_functions.h>
#include <helper_cuda.h>
using namespace nvcuda;

#define M_GLOBAL 64
#define K_GLOBAL 64
#define N_GLOBAL 64

__host__ void init_host_matrices(half *a, half *b, float *c) {
    half val = (half)(rand() % 3);
    for (int i = 0; i < M_GLOBAL; i++) {
        for (int j = 0; j < K_GLOBAL; j++) {
            a[i*K_GLOBAL+j] = (half)(rand() % 3);
        }
    }

    for (int i = 0; i < N_GLOBAL; i++) {
        for (int j = 0; j < K_GLOBAL; j++) {
            b[i*K_GLOBAL+j] = (half)(rand() % 3);
        }
    }

    for (int t = 0; t < M_GLOBAL * N_GLOBAL; t++) {
        c[t] =  0.0;//(float)(rand() % 3);
    }
}

__host__ void matMultiplyOnHost(half *A, half *B, float *C, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float temp = 0.0;

            for (int k = 0; k < K; k++) {
                temp += (float)A[i * K + k] * (float)B[j * N + k];//  (float)B[j * numBRows + k];
            }

            C[i * N + j] = temp;
        }
    }
}

__global__ void wmma_ker(half *a, half *b, float *c) {
    // Declare the fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;

    // Initialize the output to zero
    wmma::fill_fragment(acc_frag, 0.0f);

    int lda = 64;
    int ldb = 64;
    int ldc = 64;
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    for (int i = 0; i < 64; i += 16) {
        int aCol = i;
        int aRow = warpM * 16;

        int bCol = i;
        int bRow = warpN * 16;

        // B 是 col_major 模式， 所以，  A 的列 和 B 的列 相等， 为 0， 16， 32， 48
        // A， B 的行 也是 16 的整数倍， 用 x 表示 A 的行， y 表示 B 的行， 进行组合， 共 4x4 16种组合
        //     即是 A ， B 矩阵 16x16 块的左上角的坐标
        // 所以 16 种 行组合  x  4 种列组合 = 64  种组合， 即 64 个 warp 完成整个 64x64x64 的矩阵乘
        // warp 里的每个 thread 执行的 code 是一样的   2023.5.4

        // Bounds checking
        if (aRow < 64 && aCol < 64 && bRow < 64 && bCol < 64) {
            printf("warpM=%d, warpN=%d, x=%d, y=%d, i = %d, aRow = %d, aCol = %d, bRow = %d, bCol = %d\n", warpM, warpN,
                   blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y, i, aRow, aCol, bRow, bCol);
            // Load the inputs
            wmma::load_matrix_sync(a_frag, a + aCol + aRow * lda, lda);
            wmma::load_matrix_sync(b_frag, b + bCol + bRow * ldb, ldb);

            // Perform the matrix multiplication
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // Load in the current value of c
    int cCol = warpN * 16;  // 即 bRow
    int cRow = warpM * 16;  // 即 ARow

    // 每个 thread 完成 C 的一个点的拷贝， 这里结果 acc_flag 和 c 都是 row_major 模式，只是 B 是列模式方便计算
    if (cRow < 64 && cCol < 64) {
        wmma::store_matrix_sync(c + cCol + cRow * ldc, acc_frag, ldc, wmma::mem_row_major);
    }
}

int main(int argc, char **argv) {
    half *A_h = NULL;
    half *B_h = NULL;
    float *C_h = NULL;
    float *result_hD   = (float*) malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);

    A_h = (half*) malloc(sizeof(half) * M_GLOBAL * K_GLOBAL);
    B_h = (half*) malloc(sizeof(half) * K_GLOBAL * N_GLOBAL);
    C_h = (float*) malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);

    half *A = NULL;
    half *B = NULL;
    float *C = NULL;

    checkCudaErrors(cudaMalloc((void**)&A, sizeof(half) * M_GLOBAL * K_GLOBAL));
    checkCudaErrors(cudaMalloc((void**)&B, sizeof(half) * N_GLOBAL * K_GLOBAL));
    checkCudaErrors(cudaMalloc((void**)&C, sizeof(float) * M_GLOBAL * N_GLOBAL));

    init_host_matrices(A_h, B_h, C_h);
    matMultiplyOnHost(A_h, B_h, C_h, M_GLOBAL, K_GLOBAL, N_GLOBAL);

    checkCudaErrors(cudaMemcpy(A, A_h, sizeof(half) * M_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(B, B_h, sizeof(half) * N_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(C, C_h, sizeof(float) * M_GLOBAL * N_GLOBAL, cudaMemcpyHostToDevice));

    dim3 blockDim(32, 8);
    //dim3 gridDim((64 + (16 * blockDim.x / 32 - 1)) / (16 * blockDim.x / 32), (64 + 16 * blockDim.y - 1) / (16 * blockDim.y));
    dim3 gridDim(4, 1);

    wmma_ker<<<gridDim, blockDim>>>(A, B, C);
    checkCudaErrors(cudaMemcpy(result_hD, C, sizeof(float)*M_GLOBAL*N_GLOBAL, cudaMemcpyDeviceToHost));

    for (int i = 0; i < N_GLOBAL * M_GLOBAL; i++) {
        if (fabs(result_hD[i] - C_h[i]) > 0.1f)
            printf("mismatch i=%d result_hD=%f result_host=%f\n", i, result_hD[i], C_h[i]);
        return -1;
    }

    return 0;
}