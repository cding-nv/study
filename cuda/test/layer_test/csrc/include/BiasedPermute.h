#pragma once
#include "Add.h"
#include "utils.h"
#include <cuda_fp16.h>
#include <typeinfo>


template <typename T>
__global__ void
biasedPermute45Small(int num_attention_heads,
                     int all_head_size,
                     int attention_head_size,
                     const T* __restrict__ inp,
                     const T* __restrict__ bias,
                     T* __restrict__ out) {
    if (threadIdx.x >= attention_head_size) return;
    const int oStride2 = gridDim.x * attention_head_size;

    inp += ((blockIdx.y * gridDim.x + blockIdx.x) * blockDim.y + threadIdx.y) * all_head_size + threadIdx.x;
    bias += threadIdx.y * all_head_size + threadIdx.x;
    out += (threadIdx.y * gridDim.y + blockIdx.y) * gridDim.x * all_head_size + blockIdx.x * attention_head_size + threadIdx.x;

    for (int i = 0; i < num_attention_heads; ++i) {
        out[i * oStride2] = __ldg(&inp[i * attention_head_size]) + __ldg(&bias[i * attention_head_size]);
    }
}


template <typename T>
__global__ void
biasedPermute45Large(int all_head_size,
                     int attention_head_size,
                     const T* __restrict__ inp,
                     const T* __restrict__ bias,
                     T* __restrict__ out) {
    inp += ((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) * all_head_size + threadIdx.y * attention_head_size;
    bias += blockIdx.x * all_head_size + threadIdx.y * attention_head_size;
    out += (blockIdx.x * gridDim.z + blockIdx.z) * gridDim.y * all_head_size + (threadIdx.y * gridDim.y + blockIdx.y) * attention_head_size;

    for (int i = threadIdx.x; i < attention_head_size; i += blockDim.x) {
        out[i] = __ldg(&inp[i]) + __ldg(&bias[i]);
    }
}


// out-of-place
// [B * S * 3 * F] -> [3 * B * A * S * H]
template <typename T>
inline void
launchBiasedPermute(int batch_size,
                    int seq_len,
                    int all_head_size,
                    int num_attention_heads,
                    int attention_head_size,
                    const T* inp,
                    const T* bias,
                    T* out,
                    cudaStream_t stream = 0) {
    if (attention_head_size <= 32) {
        const dim3 dimBlock(32, 3);
        const dim3 dimGrid(seq_len, batch_size);
        biasedPermute45Small<<<dimGrid, dimBlock, 0, stream>>>(num_attention_heads, all_head_size, attention_head_size, inp, bias, out);
    }
    else {
        const dim3 dimBlock(min(1024 / num_attention_heads, 32), num_attention_heads);
        const dim3 dimGrid(3, seq_len, batch_size);
        biasedPermute45Large<<<dimGrid, dimBlock, 0, stream>>>(all_head_size, attention_head_size, inp, bias, out);
    }
}


template <typename T>
__global__ void
permute54Small(int num_attention_heads,
               int all_head_size,
               int attention_head_size,
               const T* __restrict__ inp,
               T* __restrict__ out) {
    if (threadIdx.x >= attention_head_size) return;
    const int iStride2 = gridDim.x * attention_head_size;

    inp += (threadIdx.y * gridDim.y + blockIdx.y) * gridDim.x * all_head_size + blockIdx.x * attention_head_size + threadIdx.x;
    out += ((blockIdx.y * gridDim.x + blockIdx.x) * blockDim.y + threadIdx.y) * all_head_size + threadIdx.x;

    for (int i = 0; i < num_attention_heads; ++i) {
        out[i * attention_head_size] = __ldg(&inp[i * iStride2]);
    }
}


template <typename T>
__global__ void
permute54Large(int all_head_size,
               int attention_head_size,
               const T* __restrict__ inp,
               T* __restrict__ out) {
    inp += (blockIdx.x * gridDim.z + blockIdx.z) * gridDim.y * all_head_size + (threadIdx.y * gridDim.y + blockIdx.y) * attention_head_size;
    out += ((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) * all_head_size + threadIdx.y * attention_head_size;

    for (int i = threadIdx.x; i < attention_head_size; i += blockDim.x) {
        out[i] = __ldg(&inp[i]);
    }
}


// out-of-place
// [3 * B * A * S * H] -> [B * S * 3 * F]
template <typename T>
inline void
launchBiasedPermuteBackward(int batch_size,
                            int seq_len,
                            int all_head_size,
                            int num_attention_heads,
                            int attention_head_size,
                            const T* grad,
                            T* dInp,
                            cudaStream_t stream = 0) {
    if (attention_head_size <= 32) {
        const dim3 dimBlock(32, 3);
        const dim3 dimGrid(seq_len, batch_size);
        permute54Small<<<dimGrid, dimBlock, 0, stream>>>(num_attention_heads, all_head_size, attention_head_size, grad, dInp);
    }
    else {
        const dim3 dimBlock(min(1024 / num_attention_heads, 32), num_attention_heads);
        const dim3 dimGrid(3, seq_len, batch_size);
        permute54Large<<<dimGrid, dimBlock, 0, stream>>>(all_head_size, attention_head_size, grad, dInp);
    }
}


// out = (inp + bias).view().permute(0, 2, 1, 3)
template <typename T>
class BiasedPermute {
public:
    static void
    forward(const T* inp,
            const T* bias,
            T* out,
            int batch_size,
            int seq_len,
            int all_head_size,
            int num_attention_heads,
            int attention_head_size,
            cudaStream_t stream = 0) {
        if (typeid(T) == typeid(__half) && !(attention_head_size & 1)) {
            launchBiasedPermute(batch_size,
                                seq_len,
                                all_head_size >> 1,
                                num_attention_heads,
                                attention_head_size >> 1,
                                reinterpret_cast<const __half2*>(inp),
                                reinterpret_cast<const __half2*>(bias),
                                reinterpret_cast<__half2*>(out),
                                stream);
        }
        else {
            launchBiasedPermute(batch_size,
                                seq_len,
                                all_head_size,
                                num_attention_heads,
                                attention_head_size,
                                inp,
                                bias,
                                out,
                                stream);
        }
    }

    static void
    backward(const T* grad,
             T* dInp,
             T* dBias,
             int batch_size,
             int seq_len,
             int all_head_size,
             int num_attention_heads,
             int attention_head_size,
             cudaStream_t stream = 0) {
        if (typeid(T) == typeid(__half) && !(attention_head_size & 1)) {
            launchBiasedPermuteBackward(batch_size,
                                        seq_len,
                                        all_head_size >> 1,
                                        num_attention_heads,
                                        attention_head_size >> 1,
                                        reinterpret_cast<const __half2*>(grad),
                                        reinterpret_cast<__half2*>(dInp),
                                        stream);
        }
        else {
            launchBiasedPermuteBackward(batch_size,
                                        seq_len,
                                        all_head_size,
                                        num_attention_heads,
                                        attention_head_size,
                                        grad,
                                        dInp,
                                        stream);
        }

        launchAddBiasOB(batch_size * seq_len,
                        3 * all_head_size,
                        dInp,
                        dBias,
                        stream);
    }
};

