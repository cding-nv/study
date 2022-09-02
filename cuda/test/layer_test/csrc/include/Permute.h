#pragma once
#include "utils.h"
#include <cuda_fp16.h>
#include <typeinfo>


template <typename T>
__global__ void
permute43Small(int num_attention_heads,
               int all_head_size,
               int attention_head_size,
               const T* __restrict__ inp,
               T* __restrict__ out) {
    if (threadIdx.x >= attention_head_size) return;
    const int iStride1 = gridDim.x * attention_head_size;

    inp += blockIdx.y * gridDim.x * all_head_size + blockIdx.x * attention_head_size + threadIdx.x;
    out += (blockIdx.y * gridDim.x + blockIdx.x) * all_head_size + threadIdx.x;

    for (int i = 0; i < num_attention_heads; ++i) {
        out[i * attention_head_size] = __ldg(&inp[i * iStride1]);
    }
}


template <typename T>
__global__ void
permute43Large(int all_head_size,
               int attention_head_size,
               const T* __restrict__ inp,
               T* __restrict__ out) {
    inp += blockIdx.y * gridDim.x * all_head_size + (threadIdx.y * gridDim.x + blockIdx.x) * attention_head_size;
    out += (blockIdx.y * gridDim.x + blockIdx.x) * all_head_size + threadIdx.y * attention_head_size;

    for (int i = threadIdx.x; i < attention_head_size; i += blockDim.x) {
        out[i] = __ldg(&inp[i]);
    }
}


// out-of-place
// [B * A * S * H] -> [B * S * F]
template <typename T>
inline void
launchPermute(int batch_size,
              int seq_len,
              int all_head_size,
              int num_attention_heads,
              int attention_head_size,
              const T* inp,
              T* out,
              cudaStream_t stream = 0) {
    if (attention_head_size <= 32) {
        const dim3 dimGrid(seq_len, batch_size);
        permute43Small<<<dimGrid, 32, 0, stream>>>(num_attention_heads, all_head_size, attention_head_size, inp, out);
    }
    else {
        const dim3 dimBlock(min(1024 / num_attention_heads, 32), num_attention_heads);
        const dim3 dimGrid(seq_len, batch_size);
        permute43Large<<<dimGrid, dimBlock, 0, stream>>>(all_head_size, attention_head_size, inp, out);
    }
}


template <typename T>
__global__ void
permute34Small(int num_attention_heads,
               int all_head_size,
               int attention_head_size,
               const T* __restrict__ inp,
               T* __restrict__ out) {
    if (threadIdx.x >= attention_head_size) return;
    const int oStride1 = gridDim.x * attention_head_size;

    inp += (blockIdx.y * gridDim.x + blockIdx.x) * all_head_size + threadIdx.x;
    out += blockIdx.y * gridDim.x * all_head_size + blockIdx.x * attention_head_size + threadIdx.x;

    for (int i = 0; i < num_attention_heads; ++i) {
        out[i * oStride1] = __ldg(&inp[i * attention_head_size]);
    }
}


template <typename T>
__global__ void
permute34Large(int all_head_size,
               int attention_head_size,
               const T* __restrict__ inp,
               T* __restrict__ out) {
    inp += (blockIdx.y * gridDim.x + blockIdx.x) * all_head_size + threadIdx.y * attention_head_size;
    out += blockIdx.y * gridDim.x * all_head_size + (threadIdx.y * gridDim.x + blockIdx.x) * attention_head_size;

    for (int i = threadIdx.x; i < attention_head_size; i += blockDim.x) {
        out[i] = __ldg(&inp[i]);
    }
}


// out-of-place
// [B * S * F] -> [B * A * S * H]
template <typename T>
inline void
launchPermuteBackward(int batch_size,
                      int seq_len,
                      int all_head_size,
                      int num_attention_heads,
                      int attention_head_size,
                      const T* grad,
                      T* dInp,
                      cudaStream_t stream = 0) {
    if (attention_head_size <= 32) {
        const dim3 dimGrid(seq_len, batch_size);
        permute34Small<<<dimGrid, 32, 0, stream>>>(num_attention_heads, all_head_size, attention_head_size, grad, dInp);
    }
    else {
        const dim3 dimBlock(min(1024 / num_attention_heads, 32), num_attention_heads);
        const dim3 dimGrid(seq_len, batch_size);
        permute34Large<<<dimGrid, dimBlock, 0, stream>>>(all_head_size, attention_head_size, grad, dInp);
    }
}


// out = inp.permute(0, 2, 1, 3).contiguous().view()
template <typename T>
class Permute {
public:
    static void
    forward(const T* inp,
            T* out,
            int batch_size,
            int seq_len,
            int all_head_size,
            int num_attention_heads,
            int attention_head_size,
            cudaStream_t stream = 0) {
        if (typeid(T) == typeid(__half) && !(attention_head_size & 1)) {
            launchPermute(batch_size,
                          seq_len,
                          all_head_size >> 1,
                          num_attention_heads,
                          attention_head_size >> 1,
                          reinterpret_cast<const __half2*>(inp),
                          reinterpret_cast<__half2*>(out),
                          stream);
        }
        else {
            launchPermute(batch_size,
                          seq_len,
                          all_head_size,
                          num_attention_heads,
                          attention_head_size,
                          inp,
                          out,
                          stream);
        }
    }

    static void
    backward(const T* grad,
             T* dInp,
             int batch_size,
             int seq_len,
             int all_head_size,
             int num_attention_heads,
             int attention_head_size,
             cudaStream_t stream = 0) {
        if (typeid(T) == typeid(__half) && !(attention_head_size & 1)) {
            launchPermuteBackward(batch_size,
                                  seq_len,
                                  all_head_size >> 1,
                                  num_attention_heads,
                                  attention_head_size >> 1,
                                  reinterpret_cast<const __half2*>(grad),
                                  reinterpret_cast<__half2*>(dInp),
                                  stream);
        }
        else {
            launchPermuteBackward(batch_size,
                                  seq_len,
                                  all_head_size,
                                  num_attention_heads,
                                  attention_head_size,
                                  grad,
                                  dInp,
                                  stream);
        }
    }
};

