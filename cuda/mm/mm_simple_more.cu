#include <stdio.h>
#include <math.h>

#define M 32
#define N 32
#define K 32

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

// 每个thread负责一个 C(i, j), 每个线程for循环次数是K
// C(0, 0) A的第0行 乘 B的第0列
__global__ void matrixMultiply(float *A, float *B, float *C) {
    float sum = 0.0f;

    // thread(row, col) is for C(i,j)
    int row = blockIdx.y * blockDim.y + threadIdx.y; // y is for row
    int col = blockIdx.x * blockDim.x + threadIdx.x; // x is for cloumn

    if(row < M && col < N) {
        for (int i = 0; i < K; ++i) {    // K loop for a thread
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    int i = 0;
    int j = 0;
    float A[M * K];
    float B[K * N];
    float C[M * N] = {0};

    for (i = 0; i < M; i++) {
        for (j = 0; j < K; j++) {
            A[i * K + j] = 1;//sin(i);
        }
    }
    for (i = 0; i < K; i++) {
        for (j = 0; j < N; j++) {
            B[i * N + j] = 1;//cos(j);
        }
    }
    //matrixMulCPU(C, A, B);
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

    dim3 dim_block(16, 16, 1);
    dim3 dim_grid((N - 1) / dim_block.x + 1, (M - 1) / dim_block.y + 1, 1);

    matrixMultiply<<<dim_grid, dim_block>>>(d_a, d_b, d_c);
    cudaMemcpy(C, d_c, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            printf(" %f ", C[i * K + j]);
        }
        printf("\n");
    }

    return 0;
}
