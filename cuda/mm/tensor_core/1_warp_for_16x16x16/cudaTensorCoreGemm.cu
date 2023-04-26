#include <assert.h>
#include <stdio.h>
#include <cuda.h>
#include <mma.h>
#include <helper_functions.h>
#include <helper_cuda.h>
using namespace nvcuda;

#define M_GLOBAL 16
#define K_GLOBAL 16
#define N_GLOBAL 16

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
                temp += (float)A[i * K + k] * (float)B[k * N + j];//  (float)B[j * numBRows + k];
            }

            C[i * N + j] = temp;
        }
    }
}

__global__ void wmma_ker(half *a, half *b, float *c) {

    printf("x=%d, y=%d\n",
                blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
   // Declare the fragments
   wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
   wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
   wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

   // Initialize the output to zero
   wmma::fill_fragment(c_frag, 0.0f);

   // Load the inputs
   wmma::load_matrix_sync(a_frag, a, 16);
   wmma::load_matrix_sync(b_frag, b, 16);

   // Perform the matrix multiplication
   wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

   // Store the output
   wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_row_major);
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
    //float *D = NULL;

    checkCudaErrors(cudaMalloc((void**)&A, sizeof(half) * M_GLOBAL * K_GLOBAL));
    checkCudaErrors(cudaMalloc((void**)&B, sizeof(half) * N_GLOBAL * K_GLOBAL));
    checkCudaErrors(cudaMalloc((void**)&C, sizeof(float) * M_GLOBAL * N_GLOBAL));
    //checkCudaErrors(cudaMalloc((void**)&D, sizeof(float) * M_GLOBAL * N_GLOBAL));

    init_host_matrices(A_h, B_h, C_h);
    matMultiplyOnHost(A_h, B_h, C_h, M_GLOBAL, K_GLOBAL, N_GLOBAL);

    checkCudaErrors(cudaMemcpy(A, A_h, sizeof(half) * M_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(B, B_h, sizeof(half) * N_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(C, C_h, sizeof(float) * M_GLOBAL * N_GLOBAL, cudaMemcpyHostToDevice));
    //checkCudaErrors(cudaMemset(D, 0, sizeof(float) * M_GLOBAL * N_GLOBAL));

    wmma_ker<<<1, 32>>>(A, B, C);
    checkCudaErrors(cudaMemcpy(result_hD, C, sizeof(float)*M_GLOBAL*N_GLOBAL, cudaMemcpyDeviceToHost));

    for (int i = 0; i < N_GLOBAL * M_GLOBAL; i++) {
        if (fabs(result_hD[i] - C_h[i]) > 0.1f)
            printf("mismatch i=%d result_hD=%f result_host=%f\n", i, result_hD[i], C_h[i]);
    }

    return 0;
}