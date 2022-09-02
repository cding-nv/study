#include "cuda_runtime.h"
#include <iostream> 

#if 0
__global__ void reduceInterleaved (int *g_idata, int *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    if(idx >= n) return;

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}
#endif

__global__ void reduceUnrolling2 (int *g_idata, int *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 2;

    // unrolling 2
    if (idx + blockDim.x < n) g_idata[idx] += g_idata[idx + blockDim.x];

    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}


#define BLOCK 2

int main() {
    int h[] = {13, 27, 15, 14, 33, 2, 24, 6};
    const auto count = sizeof(h) / sizeof(h[0]);
    const int size = count * sizeof(int);

    int* d;

    cudaMalloc(&d, size);
    cudaMemcpy(d, h, size, cudaMemcpyHostToDevice);

    int* o_d;
    cudaMalloc(&o_d, BLOCK * sizeof(int));

    reduceUnrolling2 <<<BLOCK, count / BLOCK >>>(d, o_d, size);

    int result[BLOCK];
    cudaMemcpy(&result, o_d, BLOCK * sizeof(int), cudaMemcpyDeviceToHost);

    int sum = 0;
    for (int i = 0; i < BLOCK; i++) {
        sum += result[i];
        std::cout << "block " << i << " = " << result[i] << std::endl;

    }
    std::cout << "Sum = " << result[0] + result[1] << std::endl;


    //getchar();

    cudaFree(d);
    //delete[] h;

    return 0;
}

