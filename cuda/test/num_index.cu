#include <stdio.h>

#define N 32

__global__ void kernel_index(int* input, int count, int* output) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    //input[tid] = tid * 2 - 1;
    int i = 0;
    for (i = 0; i < N; i++) {
        if (input[tid + i] == tid) {
            output[tid] = tid + i;
            break;
        }
    }

}

int main() {

    int a_h[N] = {0};
    int r_h[N] = {0};

    int i = 0;

    for (i = 0; i < N; i++) {
        a_h[i] = i % 2 == 0 ? i : i + 1;
        a_h[i] /= 2;
        printf(" %d ", a_h[i]);
    }
    printf("\n");
 
    int *data_d;
    int *out_d;
    cudaMalloc((void**)&data_d, sizeof(int) * N);
    cudaMalloc((void**)&out_d, sizeof(int) * N);
    cudaMemcpy(data_d, a_h, sizeof(int) * N, cudaMemcpyHostToDevice);
    
    kernel_index<<<1, N>>>(data_d, N, out_d);
    cudaMemcpy(r_h, out_d, sizeof(int) * N, cudaMemcpyDeviceToHost);

    for (i = 0; i < N; i++) {
        printf(" %d ", r_h[i]);
    }

    return 0;
}
