#include "AdjMatrixBatch.h"
#include <cuda_fp16.h>


template <typename T>
__global__ void
adjMatrixBatch(int seq_len,
               float alpha,
               const T* __restrict__ inp,
               T* __restrict__ out) {   // float
    inp += blockIdx.x * seq_len;
    out += blockIdx.x * seq_len * seq_len;
    extern __shared__ char _sm[];
    T* sm = reinterpret_cast<T*>(_sm);

    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        sm[i] = __ldg(&inp[i]);
    }
    __syncthreads();

    for (int j = 0; j < seq_len; ++j) {
        for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
            out[i] = T(sm[i] == sm[j] && sm[i]) * T(alpha);
        }
        out += seq_len;
    }
}


// dtypes of inp & out are required to be identical by DLRouter
template <typename T>
void
launchAdjMatrixBatch(cudaStream_t stream,
                     const T* inp,
                     T* out,
                     int batch_size,
                     int seq_len,
                     float alpha) {
    const int smem = sizeof(T) * seq_len;
    adjMatrixBatch<<<batch_size, 256, smem, stream>>>(seq_len, alpha, inp, out);
}


template
void
launchAdjMatrixBatch(cudaStream_t stream,
                     const float* inp,
                     float* out,
                     int batch_size,
                     int seq_len,
                     float alpha);


// this instantiation will not be invoked
template
void
launchAdjMatrixBatch(cudaStream_t stream,
                     const __half* inp,
                     __half* out,
                     int batch_size,
                     int seq_len,
                     float alpha);

