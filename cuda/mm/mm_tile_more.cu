#include <stdio.h>
#include <math.h>

#define M 16
#define N 16
#define K 16

#define OPT_MOVE_K 0

#if OPT_MOVE_K
void matrixMulCPU(float* C, const float* A, const float* B) {
    printf("OPT_MOVE_K\n");
    for (int k = 0; k < K; k++) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}
#else
void matrixMulCPU(float* C, const float* A, const float* B) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < K; k++) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}
#endif

#define TILE_WIDTH 8  // block size

__global__ void matrixMultiplyShared(float *A, float *B, float *C) {
    __shared__ float sharedM[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sharedN[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // (by, bx)   (0, 0)  (0, 1)
    //            (1, 0)  (1, 1)
    // 每个 block 里每个tile 的坐标（ty， tx）
    int row = by * TILE_WIDTH + ty;  // 行方向第 by 个 block
    int col = bx * TILE_WIDTH + tx;  // 列方向第 bx 个 block
    float v = 0.0;

    // 每个线程负责A B tile 中一个元素拷贝. 比如 blocksize = 16*16, 一个 block 里面有 16 * 16 个线程
    // 每个 block 可以填满需要用到的 tile 16*16 大小的矩阵, 保存在 sharedM，sharedN 中
    for (int i = 0; i < (int)(ceil((float)K / TILE_WIDTH)); i++) {
        if (i * TILE_WIDTH + tx < K && row < M)
            sharedM[ty][tx] = A[row * K + i * TILE_WIDTH + tx];
        else
            sharedM[ty][tx] = 0.0;

        if (i * TILE_WIDTH + ty < K && col < N)
            //sharedN[ty][tx] = B[(i * TILE_WIDTH + ty) * N + col]; // Global Mem Not coalesced
            sharedN[ty][tx] = B[col * N + i * TILE_WIDTH + ty];
        else
            sharedN[ty][tx] = 0.0;

        // 同步使得矩阵Ａ，和矩阵Ｂ的 tile*tile的数据保存完毕
        __syncthreads();

        // 每个线程负责计算一个A子块的fragment 和 B子块的fragment 的乘积，总共 16x16 个， for 循环 K/TILE_WIDTH 次，并累加得到 A 整行和 B 整列的乘加
        for(int j = 0; j < TILE_WIDTH; j++)
            //v += sharedM[ty][j] * sharedN[j][tx]; // 从这里看出 blocksize 要== Tile width
            v += sharedM[ty][j] * sharedN[tx][j]; //
        __syncthreads();
    }
    // 每个 thread 完成 C 的一个点，每个 block 的线程完成 C 的一个 tile
    if (row < M && col < N)
        C[row * N + col] = v;
}

int main() {
    int i = 0;
    int j = 0;
    float A[M * K];
    float B[K * N];
    float C[M * N] = {0};

    for (i = 0; i < M; i++) {
        for (j = 0; j < K; j++) {
            A[i * K + j] = sin(i);
        }
    }
    for (i = 0; i < K; i++) {
        for (j = 0; j < N; j++) {
            B[i * N + j] = cos(j);
        }
    }
    matrixMulCPU(C, A, B);
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            printf(" %f ", C[i * K + j]);
        }
        printf("\n");
    }

    printf("######### start cuda ######\n");

    float *d_a;
    float *d_b;
    float *d_c;
    cudaMalloc((void**)&d_a,M * K * sizeof(float));
    cudaMalloc((void**)&d_b,K * N * sizeof(float));
    cudaMalloc((void**)&d_c,M * N * sizeof(float));

    cudaMemcpy(d_a, A, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B, sizeof(float) * K * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, C, sizeof(float) * M * N, cudaMemcpyHostToDevice);

    dim3 dim_block(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dim_grid((N - 1) / dim_block.x + 1, (M - 1) / dim_block.y + 1, 1);

    matrixMultiplyShared<<<dim_grid, dim_block>>>(d_a, d_b, d_c);
    cudaMemcpy(C, d_c, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            printf(" %f ", C[i * K + j]);
        }
        printf("\n");
    }

    return 0;
}
