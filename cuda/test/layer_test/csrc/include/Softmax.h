#pragma once
#include "errMsg.h"
#include "utils.h"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;


template <typename T, int tileSize, int iterCnt>
__global__ void
softmaxSmall(T* data,
             int seq_len,
             int N) {
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<tileSize>(block);

    int offset = (threadIdx.x + blockIdx.x * blockDim.x) / tileSize * seq_len;
    if (offset >= N) return;
    data += offset;

    float buff[iterCnt];
    float maxi = -MAXF;
    for (int it = 0; it < iterCnt; ++it) {
        int idx = tile.thread_rank() + it * tileSize;
        if (idx >= seq_len) break;
        buff[it] = data[idx];
        maxi = fmaxf(maxi, buff[it]);
    }
    for (int i = 1; i < tileSize; i <<= 1) {
        float temp = tile.shfl_xor(maxi, i);
        maxi = fmaxf(maxi, temp);
    }

    float sum = 0.f;
    for (int it = 0; it < iterCnt; ++it) {
        if (tile.thread_rank() + it * tileSize >= seq_len) break;
        sum += buff[it] = expf(buff[it] - maxi);
    }
    for (int i = 1; i < tileSize; i <<= 1) {
        sum += tile.shfl_xor(sum, i);
    }
    sum = 1. / (sum + 1e-6);

    for (int it = 0; it < iterCnt; ++it) {
        int idx = tile.thread_rank() + it * tileSize;
        if (idx >= seq_len) break;
        data[idx] = buff[it] * sum;
    }
}


template <typename T, int iterCnt>
__global__ void
softmaxLarge(T* data,
             int seq_len) {
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);

    data += blockIdx.x * seq_len;
    float buff[iterCnt];
    extern __shared__ float sm[];

    unsigned int wid = threadIdx.x >> 5;
    unsigned int nWarp = blockDim.x >> 5;

    float maxi = -MAXF;
    for (int it = 0; it < iterCnt; ++it) {
        int idx = threadIdx.x + it * blockDim.x;
        if (idx >= seq_len) break;
        buff[it] = data[idx];
        maxi = fmaxf(maxi, buff[it]);
    }
    for (int i = 1; i < 32; i <<= 1) {
        float temp = tile.shfl_xor(maxi, i);
        maxi = fmaxf(maxi, temp);
    }
    if (tile.thread_rank() == 0) {
        sm[wid] = maxi;
    }
    block.sync();
    if (tile.thread_rank() < nWarp) {
        maxi = sm[tile.thread_rank()];
    }
    block.sync();
    for (int i = 1; i < nWarp; i <<= 1) {
        float temp = tile.shfl_xor(maxi, i);
        maxi = fmaxf(maxi, temp);
    }
    maxi = tile.shfl(maxi, 0);

    float sum = 0.f;
    for (int it = 0; it < iterCnt; ++it) {
        if (threadIdx.x + it * blockDim.x >= seq_len) break;
        sum += buff[it] = expf(buff[it] - maxi);
    }
    for (int i = 1; i < 32; i <<= 1) {
        sum += tile.shfl_xor(sum, i);
    }
    if (tile.thread_rank() == 0) {
        sm[wid] = sum;
    }
    block.sync();
    if (tile.thread_rank() < nWarp) {
        sum = sm[tile.thread_rank()];
    }
    for (int i = 1; i < nWarp; i <<= 1) {
        sum += tile.shfl_xor(sum, i);
    }
    sum = tile.shfl(sum, 0);
    sum = 1. / (sum + 1e-6);

    for (int it = 0; it < iterCnt; ++it) {
        int idx = threadIdx.x + it * blockDim.x;
        if (idx >= seq_len) break;
        data[idx] = buff[it] * sum;
    }
}


// inplace softmax(dim=-1) forward
template <typename T>
void
launchSoftmax(T* data,
              int batch_head_seq,
              int seq_len,
              cudaStream_t stream = 0) {
    int N = batch_head_seq * seq_len;

    if (seq_len <= 2) {
        softmaxSmall<T, 2, 1><<<ceilDiv(batch_head_seq, 64), 128, 0, stream>>>(data, seq_len, N);
    }
    else if (seq_len <= 4) {
        softmaxSmall<T, 4, 1><<<ceilDiv(batch_head_seq, 32), 128, 0, stream>>>(data, seq_len, N);
    }
    else if (seq_len <= 8) {
        softmaxSmall<T, 8, 1><<<ceilDiv(batch_head_seq, 16), 128, 0, stream>>>(data, seq_len, N);
    }
    else if (seq_len <= 16) {
        softmaxSmall<T, 16, 1><<<ceilDiv(batch_head_seq, 8), 128, 0, stream>>>(data, seq_len, N);
    }
    else if (seq_len <= 32) {
        softmaxSmall<T, 32, 1><<<ceilDiv(batch_head_seq, 4), 128, 0, stream>>>(data, seq_len, N);
    }
    else if (seq_len <= 64) {
        softmaxSmall<T, 32, 2><<<ceilDiv(batch_head_seq, 4), 128, 0, stream>>>(data, seq_len, N);
    }
    else if (seq_len <= 128) {
        softmaxSmall<T, 32, 4><<<ceilDiv(batch_head_seq, 4), 128, 0, stream>>>(data, seq_len, N);
    }
    else if (seq_len <= 4096) {
        const int dimBlock = nextPow2(seq_len) >> 2;
        const int smem = sizeof(float) * (dimBlock >> 5);
        softmaxLarge<T, 4><<<batch_head_seq, dimBlock, smem, stream>>>(data, seq_len);
    }
}


template <typename T, int tileSize, int iterCnt>
__global__ void
maskedSoftmaxSmall(int line,
                   int seq_len,
                   const T* mask,
                   T* data) {
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<tileSize>(block);

    int offset = blockIdx.y * line + (blockIdx.x * blockDim.x + threadIdx.x) / tileSize * seq_len;
    if (offset >= gridDim.y * line) return;
    mask += blockIdx.y * seq_len;
    data += offset;

    float buff[iterCnt];
    float maxi = -MAXF;
    for (int it = 0; it < iterCnt; ++it) {
        int idx = tile.thread_rank() + it * tileSize;
        if (idx >= seq_len) break;
        buff[it] = mask[idx] + data[idx];
        maxi = fmaxf(maxi, buff[it]);
    }
    for (int i = 1; i < tileSize; i <<= 1) {
        float temp = tile.shfl_xor(maxi, i);
        maxi = fmaxf(maxi, temp);
    }

    float sum = 0.f;
    for (int it = 0; it < iterCnt; ++it) {
        if (tile.thread_rank() + it * tileSize >= seq_len) break;
        sum += buff[it] = expf(buff[it] - maxi);
    }
    for (int i = 1; i < tileSize; i <<= 1) {
        sum += tile.shfl_xor(sum, i);
    }
    sum = 1. / (sum + 1e-6);

    for (int it = 0; it < iterCnt; ++it) {
        int idx = tile.thread_rank() + it * tileSize;
        if (idx >= seq_len) break;
        data[idx] = buff[it] * sum;
    }
}


template <typename T, int iterCnt>
__global__ void
maskedSoftmaxLarge(int seq_len,
                   const T* mask,
                   T* data) {
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);

    mask += blockIdx.y * seq_len;
    data += (blockIdx.y * gridDim.x + blockIdx.x) * seq_len;
    float buff[iterCnt];
    extern __shared__ float sm[];

    unsigned int wid = threadIdx.x >> 5;
    unsigned int nWarp = blockDim.x >> 5;

    float maxi = -MAXF;
    for (int it = 0; it < iterCnt; ++it) {
        int idx = threadIdx.x + it * blockDim.x;
        if (idx >= seq_len) break;
        buff[it] = mask[idx] + data[idx];
        maxi = fmaxf(maxi, buff[it]);
    }
    for (int i = 1; i < 32; i <<= 1) {
        float temp = tile.shfl_xor(maxi, i);
        maxi = fmaxf(maxi, temp);
    }
    if (tile.thread_rank() == 0) {
        sm[wid] = maxi;
    }
    block.sync();
    if (tile.thread_rank() < nWarp) {
        maxi = sm[tile.thread_rank()];
    }
    block.sync();
    for (int i = 1; i < nWarp; i <<= 1) {
        float temp = tile.shfl_xor(maxi, i);
        maxi = fmaxf(maxi, temp);
    }
    maxi = tile.shfl(maxi, 0);

    float sum = 0.f;
    for (int it = 0; it < iterCnt; ++it) {
        if (threadIdx.x + it * blockDim.x >= seq_len) break;
        sum += buff[it] = expf(buff[it] - maxi);
    }
    for (int i = 1; i < 32; i <<= 1) {
        sum += tile.shfl_xor(sum, i);
    }
    if (tile.thread_rank() == 0) {
        sm[wid] = sum;
    }
    block.sync();
    if (tile.thread_rank() < nWarp) {
        sum = sm[tile.thread_rank()];
    }
    for (int i = 1; i < nWarp; i <<= 1) {
        sum += tile.shfl_xor(sum, i);
    }
    sum = tile.shfl(sum, 0);
    sum = 1. / (sum + 1e-6);

    for (int it = 0; it < iterCnt; ++it) {
        int idx = threadIdx.x + it * blockDim.x;
        if (idx >= seq_len) break;
        data[idx] = buff[it] * sum;
    }
}


// inplace softmax(dim=-1) forward
template <typename T>
inline void
launchMaskedSoftmax(int batch_size,
                    int head_seq,
                    int seq_len,
                    const T* mask,
                    T* data,
                    cudaStream_t stream = 0) {
    if (seq_len <= 2) {
        maskedSoftmaxSmall<T, 2, 1><<<dim3(ceilDiv(head_seq, 64), batch_size), 128, 0, stream>>>(head_seq * seq_len, seq_len, mask, data);
    }
    else if (seq_len <= 4) {
        maskedSoftmaxSmall<T, 4, 1><<<dim3(ceilDiv(head_seq, 32), batch_size), 128, 0, stream>>>(head_seq * seq_len, seq_len, mask, data);
    }
    else if (seq_len <= 8) {
        maskedSoftmaxSmall<T, 8, 1><<<dim3(ceilDiv(head_seq, 16), batch_size), 128, 0, stream>>>(head_seq * seq_len, seq_len, mask, data);
    }
    else if (seq_len <= 16) {
        maskedSoftmaxSmall<T, 16, 1><<<dim3(ceilDiv(head_seq, 8), batch_size), 128, 0, stream>>>(head_seq * seq_len, seq_len, mask, data);
    }
    else if (seq_len <= 32) {
        maskedSoftmaxSmall<T, 32, 1><<<dim3(ceilDiv(head_seq, 4), batch_size), 128, 0, stream>>>(head_seq * seq_len, seq_len, mask, data);
    }
    else if (seq_len <= 64) {
        maskedSoftmaxSmall<T, 32, 2><<<dim3(ceilDiv(head_seq, 4), batch_size), 128, 0, stream>>>(head_seq * seq_len, seq_len, mask, data);
    }
    else if (seq_len <= 128) {
        maskedSoftmaxSmall<T, 32, 4><<<dim3(ceilDiv(head_seq, 4), batch_size), 128, 0, stream>>>(head_seq * seq_len, seq_len, mask, data);
    }
    else if (seq_len <= 4096) {
        const int dimBlock = nextPow2(seq_len) >> 2;
        const int smem = sizeof(float) * (dimBlock >> 5);
        maskedSoftmaxLarge<T, 4><<<dim3(head_seq, batch_size), dimBlock, smem, stream>>>(seq_len, mask, data);
    }
}


template <typename T, int tileSize, int iterCnt>
__global__ void
softmaxBackwardSmall(T* grad,
                     const T* output,
                     int seq_len,
                     int N) {
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<tileSize>(block);

    int offset = (threadIdx.x + blockIdx.x * blockDim.x) / tileSize * seq_len;
    if (offset >= N) return;
    grad += offset;
    output += offset;

    float buffG[iterCnt];
    float buffO[iterCnt];
    float sum = 0.f;
    for (int it = 0; it < iterCnt; ++it) {
        int idx = tile.thread_rank() + it * tileSize;
        if (idx >= seq_len) break;
        buffG[it] = grad[idx];
        buffO[it] = output[idx];
        sum += buffG[it] * buffO[it];
    }
    for (int i = 1; i < tileSize; i <<= 1) {
        sum += tile.shfl_xor(sum, i);
    }

    for (int it = 0; it < iterCnt; ++it) {
        int idx = tile.thread_rank() + it * tileSize;
        if (idx >= seq_len) break;
        grad[idx] = buffO[it] * (buffG[it] - sum);
    }
}


template <typename T, int iterCnt>
__global__ void
softmaxBackwardLarge(T* grad,
                     const T* output,
                     int seq_len) {
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);

    grad += blockIdx.x * seq_len;
    output += blockIdx.x * seq_len;
    float buffG[iterCnt];
    float buffO[iterCnt];
    extern __shared__ float sm[];

    unsigned int wid = threadIdx.x >> 5;
    unsigned int nWarp = blockDim.x >> 5;

    float sum = 0.f;
    for (int it = 0; it < iterCnt; ++it) {
        int idx = threadIdx.x + it * blockDim.x;
        if (idx >= seq_len) break;
        buffG[it] = grad[idx];
        buffO[it] = output[idx];
        sum += buffG[it] * buffO[it];
    }
    for (int i = 1; i < 32; i <<= 1) {
        sum += tile.shfl_xor(sum, i);
    }
    if (tile.thread_rank() == 0) {
        sm[wid] = sum;
    }
    block.sync();
    if (tile.thread_rank() < nWarp) {
        sum = sm[tile.thread_rank()];
    }
    for (int i = 1; i < nWarp; i <<= 1) {
        sum += tile.shfl_xor(sum, i);
    }
    sum = tile.shfl(sum, 0);

    for (int it = 0; it < iterCnt; ++it) {
        int idx = threadIdx.x + it * blockDim.x;
        if (idx >= seq_len) break;
        grad[idx] = buffO[it] * (buffG[it] - sum);
    }
}


// inplace softmax(dim=-1) backward
template <typename T>
void
launchSoftmaxBackward(const T* output,          // softmax forward output
                      T* grad,
                      int batch_head_seq,
                      int seq_len,
                      cudaStream_t stream = 0) {
    int N = batch_head_seq * seq_len;

    if (seq_len <= 2) {
        softmaxBackwardSmall<T, 2, 1><<<ceilDiv(batch_head_seq, 64), 128, 0, stream>>>(grad, output, seq_len, N);
    }
    else if (seq_len <= 4) {
        softmaxBackwardSmall<T, 4, 1><<<ceilDiv(batch_head_seq, 32), 128, 0, stream>>>(grad, output, seq_len, N);
    }
    else if (seq_len <= 8) {
        softmaxBackwardSmall<T, 8, 1><<<ceilDiv(batch_head_seq, 16), 128, 0, stream>>>(grad, output, seq_len, N);
    }
    else if (seq_len <= 16) {
        softmaxBackwardSmall<T, 16, 1><<<ceilDiv(batch_head_seq, 8), 128, 0, stream>>>(grad, output, seq_len, N);
    }
    else if (seq_len <= 32) {
        softmaxBackwardSmall<T, 32, 1><<<ceilDiv(batch_head_seq, 4), 128, 0, stream>>>(grad, output, seq_len, N);
    }
    else if (seq_len <= 64) {
        softmaxBackwardSmall<T, 32, 2><<<ceilDiv(batch_head_seq, 4), 128, 0, stream>>>(grad, output, seq_len, N);
    }
    else if (seq_len <= 128) {
        softmaxBackwardSmall<T, 32, 4><<<ceilDiv(batch_head_seq, 4), 128, 0, stream>>>(grad, output, seq_len, N);
    }
    else if (seq_len <= 4096) {
        const int dimBlock = nextPow2(seq_len) >> 2;
        const int smem = sizeof(float) * (dimBlock >> 5);
        softmaxBackwardLarge<T, 4><<<batch_head_seq, dimBlock, smem, stream>>>(grad, output, seq_len);
    }
}


template <typename T>
class Softmax {
public:
    // out = softmax(dim=-1)(inp)
    static void
    forward(T* data,
            int batch_head_seq,
            int seq_len,
            cudaStream_t stream = 0) {
#ifdef _DEBUG
        if (seq_len > 4096) {
            errMsg(format("Unsupported sequence length (%d) for Softmax", seq_len));
        }
#endif
        launchSoftmax(data,
                      batch_head_seq,
                      seq_len,
                      stream);
    }

    // out = softmax(dim=-1)(inp + mask)
    static void
    forwardM(const T* mask,
             T* data,
             int batch_size,
             int seq_len,
             int num_attention_heads,
             cudaStream_t stream = 0) {
#ifdef _DEBUG
        if (seq_len > 4096) {
            errMsg(format("Unsupported sequence length (%d) for MaskedSoftmax", seq_len));
        }
#endif
        launchMaskedSoftmax(batch_size,
                            num_attention_heads * seq_len,
                            seq_len,
                            mask,
                            data,
                            stream);
    }

    static void
    backward(const T* output,
             T* grad,
             int batch_head_seq,
             int seq_len,
             cudaStream_t stream = 0) {
#ifdef _DEBUG
        if (seq_len > 4096) {
            errMsg(format("Unsupported sequence length (%d) for Softmax backward", seq_len));
        }
#endif
        launchSoftmaxBackward(output,
                              grad,
                              batch_head_seq,
                              seq_len,
                              stream);
    }
};

