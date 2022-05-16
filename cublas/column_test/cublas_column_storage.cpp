#include <iostream>
#include <cstdlib>
#include <cublas_v2.h>
#include "cuda_runtime.h"

// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)
// https://blog.csdn.net/qq632544991p/article/details/49894005

void gpu_blas_mmul(const float *A, const float *B, float *C, const int m, const int k, const int n) {
    // lda is leading dimension a, means row count
    int lda = m, ldb = k, ldc = m;
    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;

    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Do the actual multiplication
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

    // Destroy the handle
    cublasDestroy(handle);
}

int main() {
    int row1=2;
    int column1 = 4;
    int row2 = 4;
    int column2 = 3;


    int result[row1][column2];

    float *h_A = (float *)malloc(row1 * column1 * sizeof(float));
    float *h_B = (float *)malloc(row2 * column2 * sizeof(float));
    float *h_C = (float *)malloc(row1 * column2 * sizeof(float));

    for(int i = 0; i < row1; i++) {
        for(int j = 0; j < column1; j++)
            h_A[i * column1 + j] = i + 1;
    }

    for(int i = 0; i < row2; i++) {
        for(int j = 0; j < column2; j++)
            h_B[i * column2 + j] = j + 1;
    }

    for(int i=0; i < row1; i++) {
        for(int j = 0; j< column1; j++)
            printf("%.3f ",h_A[i * column1 + j]);
        printf("\n");
    }

    for(int i=0; i < row2; i++) {
        for(int j=0; j < column2; j++)
            printf("%.3f ",h_B[i * column2 + j]);
        printf("\n");
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, row1 * column1 * sizeof(float));
    cudaMalloc(&d_B, row2 * column2 * sizeof(float));
    cudaMalloc(&d_C, row1 * column2 * sizeof(float));

    // If you already have useful values in A and B you can copy them in GPU:
    cudaMemcpy(d_A, h_A, row1 * column1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, row2 * column2 * sizeof(float), cudaMemcpyHostToDevice);

    gpu_blas_mmul(d_B, d_A, d_C, row2, row1, column2);

    // Copy (and print) the result on host memory
    cudaMemcpy(h_C, d_C, row1 * column2 * sizeof(float), cudaMemcpyDeviceToHost);
    for(int i = 0; i < row1; i++) {
        for(int j = 0; j < column2; j++)
            result[i][j] = (int)h_C[j * row1 + i];
    }

    for(int i=0; i<row1; i++) {
        for(int j=0; j<column2; j++)
            printf("%d ",result[i][j]);
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);
    getchar();
    return 0;
}
