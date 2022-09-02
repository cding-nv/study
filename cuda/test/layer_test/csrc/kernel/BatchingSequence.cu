#include "BatchingSequence.h"
#include "utils.h"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;


template <typename T>
__global__ void
maskLength(int seq_len,
           int sub_len,
           const T* __restrict__ inp,
           int64_t* __restrict__ length) {
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);
    extern __shared__ int sm[];
    for (int i = block.thread_rank(); i < seq_len; i += block.size()) {
        sm[i] = 0;
    }
    block.sync();

    inp += (blockIdx.x * seq_len + threadIdx.y) * sub_len;
    for (int y = threadIdx.y; y < seq_len; y += blockDim.y) {
        for (int x = threadIdx.x; x < sub_len; x += blockDim.x) {
            if (inp[x]) sm[y] = 1;
            if (sm[y]) break;
        }
        if (sm[y] == 0) break;
        inp += blockDim.y * sub_len;
    }
    block.sync();

    int sumT = 0;
    for (int i = block.thread_rank(); i < seq_len; i += block.size()) {
        sumT += sm[i];
    }
    for (int i = 1; i < 32; i <<= 1) {
        sumT += tile.shfl_xor(sumT, i);
    }
    block.sync();
    if (tile.thread_rank() == 0) {
        sm[block.thread_rank() >> 5] = sumT;
    }
    block.sync();

    int nWarp = block.size() >> 5;
    if (tile.thread_rank() < nWarp) {
        sumT = sm[tile.thread_rank()];
    }
    for (int i = 1; i < nWarp; i <<= 1) {
        sumT += tile.shfl_xor(sumT, i);
    }
    if (block.thread_rank() == 0) {
        length[blockIdx.x] = sumT;
    }
}


template <typename T>
void
launchMaskLength(cudaStream_t stream,
                 const T* inp,
                 int64_t* length,
                 int batch_size,
                 int seq_len,
                 int sub_len) {
    const int dimX = min(max(1, nextPow2(sub_len) >> 2), 1024);
    const int dimY = min(nextPow2(seq_len), 1024 / dimX);
    const int nWarp = (dimX * dimY) >> 5;

    const int smem = sizeof(int) * max(seq_len, nWarp);
    const dim3 dimBlock(dimX, dimY);
    maskLength<<<batch_size, dimBlock, smem, stream>>>(seq_len, sub_len, inp, length);
}


template
void
launchMaskLength(cudaStream_t stream,
                 const int64_t* inp,
                 int64_t* length,
                 int batch_size,
                 int seq_len,
                 int sub_len);


template <typename T>
__global__ void
batchingSequence(int seq_len,
                 int sub_len,
                 const T* __restrict__ inp,
                 int64_t* __restrict__ length,
                 T* __restrict__ out) {
    auto block = cg::this_thread_block();
    extern __shared__ int sm[];
    int* dst = sm;
    int* src = sm + gridDim.x;
    for (int i = block.thread_rank(); i < gridDim.x; i += block.size()) {
        dst[i] = i ? length[i - 1] : 0;
    }
    block.sync();

    for (int s = 1; s < gridDim.x; s <<= 1) {
        int* tmp = dst;
        dst = src;
        src = tmp;
        for (int i = block.thread_rank(); i < gridDim.x; i += block.size()) {
            if (i >= s) {
                dst[i] = src[i] + src[i - s];
            }
            else {
                dst[i] = src[i];
            }
        }
        block.sync();
    }
    if (block.thread_rank() == 0) {
        length[gridDim.x + blockIdx.x] = dst[blockIdx.x];
    }

    inp += (blockIdx.x * seq_len + threadIdx.y) * sub_len;
    out += (dst[blockIdx.x] + threadIdx.y) * sub_len;
    for (int y = threadIdx.y; y < length[blockIdx.x]; y += blockDim.y) {
        for (int x = threadIdx.x; x < sub_len; x += blockDim.x) {
            out[x] = __ldg(&inp[x]);
        }
        inp += blockDim.y * sub_len;
        out += blockDim.y * sub_len;
    }
}


template <typename T>
void
launchBatchingSequence(cudaStream_t stream,
                       const T* inp,
                       int64_t* length,
                       T* out,
                       int batch_size,
                       int seq_len,
                       int sub_len) {
    const int dimX = min(nextPow2(sub_len), 1024);
    const int dimY = min(nextPow2(seq_len), 1024 / dimX);

    const int smem = sizeof(int) * batch_size * 2;
    const dim3 dimBlock(dimX, dimY);
    batchingSequence<<<batch_size, dimBlock, smem, stream>>>(seq_len, sub_len, inp, length, out);
}


template
void
launchBatchingSequence(cudaStream_t stream,
                       const int64_t* inp,
                       int64_t* length,
                       int64_t* out,
                       int batch_size,
                       int seq_len,
                       int sub_len);

