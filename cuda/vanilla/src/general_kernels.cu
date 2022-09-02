/*
* Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
/**
* Open sourced multi-head attention
**/

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <assert.h>
#include "general_kernels.h"

template <typename T>
__inline__ __device__
T warpReduceSum(T val)
{
  #pragma unroll
  for(int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  return val;
}

template <typename T>
__inline__ __device__
T warpReduceSum_t4(T val)
{
  #pragma unroll
  for(int mask = 2; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  return val;
}

template <typename T>
__inline__ __device__
T warpReduceSum_t8(T val)
{
  #pragma unroll
  for(int mask = 4; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  return val;
}

template <typename T>
__inline__ __device__
T warpReduceSum_t16(T val)
{
  #pragma unroll
  for(int mask = 8; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  return val;
}

template <typename T>
__inline__ __device__
T blockReduceSum(T val)
{
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum<T>(val);

  if(lane == 0)
    shared[wid] = val;
  __syncthreads();

  val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : (T)0.0f;
  val = warpReduceSum(val);
  return val;
}

// grid.x = batch_size * seq_len * head_num
// block.x = size_per_head (32)
// input: [batch_size, seq_len, head_num, size_per_head]
// output: [head_num, batch_size, seq_len, size_per_head]
template<typename T>
__global__
void transpose2013(const T* src, T* dst, const int batch_size, const int seq_len, const int head_num, const int size_per_head)
{
    int batch_id = blockIdx.x / (seq_len * head_num);
    int seq_id = (blockIdx.x / head_num) % seq_len;
    int head_id = blockIdx.x % head_num;

    if (threadIdx.x < size_per_head)
    {
        dst[head_id * (batch_size * seq_len * size_per_head) + batch_id * seq_len * size_per_head
            + seq_id * size_per_head + threadIdx.x] = src[blockIdx.x * size_per_head + threadIdx.x];
    }
}

template<typename T>
void launch_transpose2013(const T* src, T* dst,
                             const int batch_size,
                             const int seq_len,
                             const int head_num,
                             const int size_per_head,
                             cudaStream_t& stream);

template<>
void launch_transpose2013<float>(const float* src, float* dst,
                             const int batch_size,
                             const int seq_len,
                             const int head_num,
                             const int size_per_head,
                             cudaStream_t& stream){
    dim3 grid_dim(batch_size * seq_len * head_num, 1, 1);
    dim3 block_dim(size_per_head, 1, 1);

    transpose2013<<<grid_dim, block_dim, 0, stream>>>(src, dst,
            batch_size, seq_len, head_num, size_per_head);

}

template<>
void launch_transpose2013<half>(const half* src, half* dst,
                             const int batch_size,
                             const int seq_len,
                             const int head_num,
                             const int size_per_head,
                             cudaStream_t& stream){
    dim3 grid_dim(batch_size * seq_len * head_num, 1, 1);
    dim3 block_dim(size_per_head, 1, 1);

    transpose2013<<<grid_dim, block_dim, 0, stream>>>(src, dst,
            batch_size, seq_len, head_num, size_per_head);

}

// grid.x = batch_size * seq_len * head_num
// block.x = size_per_head (32)
// input: [head_num, batch_size, seq_len, size_per_head]
// output: [batch_size, seq_len, head_num, size_per_head]
template<typename T>
__global__
void transpose1203(const T* src, T* dst, const int batch_size, const int seq_len, const int head_num, const int size_per_head)
{
    int batch_id = (blockIdx.x / seq_len) % batch_size;
    int seq_id = blockIdx.x % seq_len;
    int head_id = blockIdx.x / (batch_size * seq_len);

    if (threadIdx.x < size_per_head)
    {
        dst[batch_id * (head_num * seq_len * size_per_head) + seq_id * head_num * size_per_head
            + head_id * size_per_head + threadIdx.x] = src[blockIdx.x * size_per_head + threadIdx.x];
    }
}

template<typename T>
void launch_transpose1203(const T* src, T* dst,
                             const int batch_size,
                             const int seq_len,
                             const int head_num,
                             const int size_per_head,
                             cudaStream_t& stream);

template<>
void launch_transpose1203<float>(const float* src, float* dst,
                             const int batch_size,
                             const int seq_len,
                             const int head_num,
                             const int size_per_head,
                             cudaStream_t& stream){
    dim3 grid_dim(batch_size * seq_len * head_num, 1, 1);
    dim3 block_dim(size_per_head, 1, 1);

    transpose1203<<<grid_dim, block_dim, 0, stream>>>(src, dst,
            batch_size, seq_len, head_num, size_per_head);

}

template<>
void launch_transpose1203<half>(const half* src, half* dst,
                             const int batch_size,
                             const int seq_len,
                             const int head_num,
                             const int size_per_head,
                             cudaStream_t& stream){
    dim3 grid_dim(batch_size * seq_len * head_num, 1, 1);
    dim3 block_dim(size_per_head, 1, 1);

    transpose1203<<<grid_dim, block_dim, 0, stream>>>(src, dst,
            batch_size, seq_len, head_num, size_per_head);

}

// grid.x = batch_size * seq_len * head_num
// block.x = size_per_head (32)
// input: [head_num, batch_size, seq_len, size_per_head]
// output: [batch_size, seq_len, head_num, size_per_head]
// alpha: alpha used in leaky relu
template<typename T>
__global__
void transpose1203_lreluback(const T* src, T* dst, const T* value_layer, const float alpha,
        const int batch_size, const int seq_len, const int head_num, const int size_per_head)
{
    int batch_id = (blockIdx.x / seq_len) % batch_size;
    int seq_id = blockIdx.x % seq_len;
    int head_id = blockIdx.x / (batch_size * seq_len);

    int in_idx = blockIdx.x * size_per_head + threadIdx.x;
    int out_idx = batch_id * (head_num * seq_len * size_per_head) + seq_id * head_num * size_per_head +
        head_id * size_per_head + threadIdx.x;

    if (threadIdx.x < size_per_head)
    {
        T t_src = src[in_idx];
        T t_layer = value_layer[in_idx];
        T tmp = 0.0f;

        if ((float)t_layer > 0.0f)
            tmp = t_src;
        else
            tmp = (T)((float)t_src * alpha);

        dst[out_idx] = tmp;
    }
}

// grid.x = batch_size * head_num
// block.x = size_per_head (32)
// input: [head_num, batch_size, seq_len, size_per_head]
// output: [batch_size, seq_len, head_num, size_per_head]
// this version will loop over the seq_len
template<typename T>
__global__
void transpose1203_lreluback_v2(const T* src, T* dst, const T* value_layer, const float alpha,
        const int batch_size, const int seq_len, const int head_num, const int size_per_head)
{
    int batch_id = blockIdx.x % batch_size;
    int head_id = blockIdx.x / batch_size;

    int in_idx_o = threadIdx.x + blockIdx.x * size_per_head * seq_len;
    int out_idx_o = threadIdx.x + head_id * size_per_head +
            batch_id * seq_len * head_num * size_per_head;

    for (int seq_id = 0; seq_id < seq_len; seq_id++) {
        int in_idx = in_idx_o + seq_id * size_per_head;
        int out_idx = out_idx_o + seq_id * size_per_head * head_num;

        if (threadIdx.x < size_per_head) {
            T t_src = src[in_idx];
            T t_layer = value_layer[in_idx];
            T tmp = 0.0f;

            if ((float)t_layer > 0.0f)
                tmp = t_src;
            else
                tmp = (T)((float)t_src * alpha);

            dst[out_idx] = tmp;
        }
    }
}

template<typename T>
void launch_transpose1203_lreluback(const T* src, T* dst,
                             const T* value_layer,
                             float alpha,
                             const int batch_size,
                             const int seq_len,
                             const int head_num,
                             const int size_per_head,
                             cudaStream_t& stream);

template<>
void launch_transpose1203_lreluback<float>(const float* src, float* dst,
                             const float* value_layer,
                             float alpha,
                             const int batch_size,
                             const int seq_len,
                             const int head_num,
                             const int size_per_head,
                             cudaStream_t& stream){
//    dim3 grid_dim(batch_size * seq_len * head_num, 1, 1);
//    dim3 block_dim(size_per_head, 1, 1);
//
//    transpose1203_lreluback<<<grid_dim, block_dim, 0, stream>>>(src, dst,
//            value_layer, alpha, batch_size, seq_len, head_num, size_per_head);

//    cudaDeviceSynchronize();
//    cudaError_t err = cudaGetLastError();
//    if (err != cudaSuccess)
//        printf("Failed before the kernel!\n");

    dim3 grid_dim(head_num * batch_size, 1, 1); // seq_len will be considered in the kernel
    dim3 block_dim(size_per_head, 1, 1);

    transpose1203_lreluback_v2<<<grid_dim, block_dim, 0, stream>>>(src, dst,
            value_layer, alpha, batch_size, seq_len, head_num, size_per_head);

//    cudaDeviceSynchronize();
//    err = cudaGetLastError();
//    if (err != cudaSuccess)
//        printf("Failed before the kernel!\n");
//
}

template<>
void launch_transpose1203_lreluback<half>(const half* src, half* dst,
                             const half* value_layer,
                             float alpha,
                             const int batch_size,
                             const int seq_len,
                             const int head_num,
                             const int size_per_head,
                             cudaStream_t& stream){
    dim3 grid_dim(batch_size * seq_len * head_num, 1, 1);
    dim3 block_dim(size_per_head, 1, 1);

    transpose1203_lreluback<<<grid_dim, block_dim, 0, stream>>>(src, dst,
            value_layer, alpha, batch_size, seq_len, head_num, size_per_head);

}

// kernel for backprop computation of query masking, softmax, key masking
// grid.x = B * N * F
// block.x = T , T <= 1024
// TODO: loop of head num for better performance ???
template <typename T>
__global__
void backprop_masking_softmax(T *d_attention, const T* tf_softmax,
        const bool* mask, T* d_score,
        int head_num, float rate, int bsz, int from_seq_len,
        int to_seq_len, int size_per_head ) {

    //int seq_id = blockIdx.x % from_seq_len;
    //int batch_id = blockIdx.x / from_seq_len % bsz;
    //int batch_id = blockIdx.x / (head_num * from_seq_len);
    //int head_id = blockIdx.x / from_seq_len / bsz % head_num;

    int index = threadIdx.x + blockIdx.x * to_seq_len;

    float da = 0.0f;
    float sfmax = 0.0f;
    bool m = true;
    if (threadIdx.x < to_seq_len) {
        da = (float)d_attention[index];
        sfmax = (float)tf_softmax[index];
        if (rate > 0.0f) {
            m = mask[index];
            da = m ? da / (1.0f - rate) : 0.0f;
        }
    }

    // k mask is not needed for backprop
//TODO  int index_k_mask = threadIdx.x + batch_id * to_seq_len;
//TODO    bool kmask = false;
//TODO    if (threadIdx.x < to_seq_len) {
//TODO        kmask = k_mask[index_k_mask];
//TODO    }

    // backward for softmax
    float sm1 = 0.0f;
    if (threadIdx.x < to_seq_len)
        sm1 = da * sfmax;

    float tmp = blockReduceSum<float>(sm1);
    __shared__ float sm2;
    if (threadIdx.x == 0)
        sm2 = (float)tmp;
    __syncthreads();

    if (threadIdx.x < to_seq_len) {
        // d_score
        tmp = sfmax * (da - sm2);

        // TODO: can be optimized by compute the scalar on CPU
        tmp = tmp * rsqrtf((float)size_per_head);

        d_score[index] = (T)tmp;
    }
}

template <typename T>
void launch_backprop_masking_softmax(T *d_attention, const T* tf_softmax,
        const bool* mask, T* d_score,
        int head_num, int bsz, int from_seq_len,
        int to_seq_len, int size_per_head, float rate,
        cudaStream_t& stream);

template <>
void launch_backprop_masking_softmax<half>(half *d_attention, const half* tf_softmax,
        const bool* mask, half* d_score,
        int head_num, int bsz, int from_seq_len,
        int to_seq_len, int size_per_head, float rate,
        cudaStream_t& stream) {

    dim3 grid_dim(bsz * head_num * from_seq_len, 1, 1);
    dim3 block_dim((to_seq_len + 31)/32 * 32, 1, 1);

    backprop_masking_softmax<<<grid_dim, block_dim, 0, stream>>> (d_attention,
            tf_softmax, mask, d_score, head_num, rate,
            bsz, from_seq_len, to_seq_len, size_per_head);
}

template <>
void launch_backprop_masking_softmax<float>(float *d_attention, const float* tf_softmax,
        const bool* mask, float* d_score,
        int head_num, int bsz, int from_seq_len,
        int to_seq_len, int size_per_head, float rate,
        cudaStream_t& stream) {

    dim3 grid_dim(bsz * head_num * from_seq_len, 1, 1);
    dim3 block_dim((to_seq_len + 31)/32 * 32, 1, 1);

    backprop_masking_softmax<<<grid_dim, block_dim, 0, stream>>> (d_attention,
            tf_softmax, mask, d_score, head_num, rate,
            bsz, from_seq_len, to_seq_len, size_per_head);
}


//TODO: h*dk0 <=1024
// grid = (N*S, 1, 1)
// block = (h*dk0), note h * dk0 < 1024
// input_b: [N, S, h*dk0]
// output_b: [h*dk0]
template <typename T>
__global__ void reduce_sum_bias(T* input_b, T* output_b, int head_num, int dk0, int batch_size, int seq_len)
{
    int hdk0 = head_num * dk0;
    int to_index = threadIdx.x;
    int from_index = threadIdx.x + blockIdx.x * hdk0;

    if (threadIdx.x < hdk0)
    {
        atomicAdd(&output_b[to_index], input_b[from_index]);
    }
}

// grid = (L, N)
// block = (h*dk0), TODO: note h * dk0 < 1024
// input_w: [N, L, h*dk0]
// output_w: [L, h*dk0]
template <typename T>
__global__ void reduce_sum_weight(T* input_w, T* output_w, int head_num, int dk0, int batch_size, int hidden_size)
{
    int hdk0 = head_num * dk0;
    int to_index = threadIdx.x + blockIdx.x * hdk0;
    int from_index = to_index + blockIdx.y * hidden_size * hdk0;

    if (threadIdx.x < hdk0) {
        atomicAdd(&output_w[to_index], input_w[from_index]);
    }
}

// [input] grad_bg: grad before gather for the view of backward flow
// [output] grad_ag: grad after gather
// dim0: h * dk0
template <typename T>
__global__ void gather_grad(T* grad_bg, const int* indices, T* grad_ag, int dim0, const int batch_m)
{
    int idx_x = threadIdx.x + blockIdx.x * dim0;
    int idx_y = blockIdx.y; // batch
    int in_id = idx_x + idx_y * gridDim.x * dim0;
    int out_id = idx_x + indices[blockIdx.y] * gridDim.x * dim0;

    if (threadIdx.x < dim0) {
        T value = grad_bg[in_id];
        atomicAdd(&grad_ag[out_id], value);
    }
}

//namespace cg = cooperative_groups;
// tf.reduce_sum(inputs, axis=0)  for w/b of query/key/value

// grid (C_max), C_max = max(C_q, C_k, C_v)
// block (C)
// TODO: optimization techniques, when C is very big, split into the following dim:
// grid(N, C_manx), block(C/N) when C is larger than 1024
template <typename T>
__global__ void reduce_sum_w(const T* __restrict__ input_q,
                             T* __restrict__ output_q,
                             int batch,
                             int C,
                             int C_q)
{
    int tid = threadIdx.x + blockIdx.x * C; // [C_q, C]

    // Compute of Query
    float sum = 0.0f;

    for (int i = 0; i < batch; i ++) {

        int input_id = tid + i * C_q * C;

        if (threadIdx.x < C && blockIdx.x < C_q) {
            T data = input_q[input_id];
            sum += (float)data;
        }
    }

//    for (int i = 0; i < batch /2 ; i++)
//    {
//        int input_id = tid + 2 * i * C_q * C;
//
//        if (threadIdx.x < C && blockIdx.x < C_q) {
//            T data0 = input_q[input_id + 0 * C_q * C];
//            T data1 = input_q[input_id + 1 * C_q * C];
//            sum += (float)data0 + (float)data1;
//        }
//    }

    if (threadIdx.x < C && blockIdx.x < C_q)
        output_q[tid] = (T)sum;
}

template <typename T>
void launch_sum_reduce_w(const T* inp_q,
                         T* out_q,
                         int batch,
                         int C,
                         int C_q,
                         cudaStream_t stream);

template <>
void launch_sum_reduce_w<float>(const float* inp_q,
                         float* out_q,
                         int batch,
                         int C,
                         int C_q,
                         cudaStream_t stream)
{
    assert (C <= 1024);

    dim3 grid_dim(C_q, 1, 1);
    dim3 block_dim((C + 31)/32 * 32);

    reduce_sum_w<float><<<grid_dim, block_dim, 0, stream>>>(
            inp_q, out_q, batch, C, C_q);
}

template <>
void launch_sum_reduce_w<half>(const half* inp_q,
                         half* out_q,
                         int batch,
                         int C,
                         int C_q,
                         cudaStream_t stream)
{
    dim3 grid_dim(C_q, 1, 1);
    dim3 block_dim((C + 31)/32 * 32);

    reduce_sum_w<half><<<grid_dim, block_dim, 0, stream>>>(
            inp_q, out_q, batch, C, C_q);
}

namespace cg = cooperative_groups;

template <typename T>
__global__ void column_sum_reduce(const T* __restrict__ inp,
                                  T* __restrict__ out,
                                  int rows,
                                  int width)
{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<TILE_DIM> g = cg::tiled_partition<TILE_DIM>(b);

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int y_stride = width * TILE_DIM;

    float localSum = 0;

    // Loop across matrix height
    if (idx < width) {
        int offset = threadIdx.y * width + idx;
        for (int r = threadIdx.y; r < rows; r += TILE_DIM) {
            localSum += (float)inp[offset];
            offset += y_stride;
        }
    }

    tile[threadIdx.x][threadIdx.y] = localSum;

    __syncthreads();

    // Sum the shared buffer.
    float sum = tile[threadIdx.y][threadIdx.x];

    __syncthreads();

    for (int i = 1; i < TILE_DIM; i <<= 1) sum += g.shfl_down(sum, i);

    if (threadIdx.x == 0) {
        int pos = blockIdx.x * TILE_DIM + threadIdx.y;
        if (pos < width) out[pos] = sum;
    }
}
template <typename T>
void launch_column_sum_reduce(const T* inp,
                                       T* out,
                                       int rows,
                                       int cols,
                                       cudaStream_t stream);

template <>
void launch_column_sum_reduce<float>(const float* inp,
                                              float* out,
                                              int rows, // rows = N*T
                                              int cols, // C
                                              cudaStream_t stream)
{
    // assert(rows % TILE_DIM == 0);
    // assert(cols % TILE_DIM == 0);

    dim3 grid_dim((cols - 1) / TILE_DIM + 1);
    dim3 block_dim(TILE_DIM, TILE_DIM);

    column_sum_reduce<float><<<grid_dim, block_dim, 0, stream>>>(inp, out, rows, cols);
}

template <>
void launch_column_sum_reduce<half>(const half* inp,
                                               half* out,
                                               int rows,
                                               int cols,
                                               cudaStream_t stream)
{
    // assert(rows % TILE_DIM == 0);
    // assert(cols % TILE_DIM == 0);

    dim3 grid_dim((cols - 1) / TILE_DIM + 1);
    dim3 block_dim(TILE_DIM, TILE_DIM);

    column_sum_reduce<half><<<grid_dim, block_dim, 0, stream>>>(inp, out, rows, cols);
}

