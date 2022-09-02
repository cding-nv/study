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
#include <iostream>
#include <fstream>
#include <curand_kernel.h>
#include "common_op.h"

namespace multiheadattention{

#define FINAL_MASK 0xffffffff
#define CUDART_PI_F 3.141592654f

template <typename T>
__inline__ __device__
T gelu(T x)
{
  float cdf = 0.5f * (1.0f + tanhf((0.7978845608028654f * (x + 0.044715f * x * x * x))));
  return x * cdf;
}

template <>
__inline__ __device__
half2 gelu(half2 val)
{
  half2 val_pow3 = __hmul2(val, __hmul2(val, val));
  float2 tmp_pow = __half22float2(val_pow3);
  float2 tmp =  __half22float2(val);

  tmp.x = 0.5f * (1.0f + tanhf((0.7978845608028654f * (tmp.x + 0.044715f * tmp_pow.x))));
  tmp.y = 0.5f * (1.0f + tanhf((0.7978845608028654f * (tmp.y + 0.044715f * tmp_pow.y))));
  return __hmul2(val, __float22half2_rn(tmp));
}

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

template <typename T>
  __inline__ __device__
T warpReduceMax(T val)
{
  #pragma unroll
  for(int mask = 16; mask > 0; mask >>= 1)
    val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
  return val;
}

/* Calculate the maximum of all elements in a block */
template <typename T>
  __inline__ __device__
T blockReduceMax(T val)
{
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f; // in-warp idx
  int wid = threadIdx.x >> 5;  // warp idx

  val = warpReduceMax(val); // get maxx in each warp

  if(lane == 0) // record in-warp maxx by warp Idx
    shared[wid] = val;

  __syncthreads();


  val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : (T)-1e20f;
  val = warpReduceMax<T>(val);

  return val;
}


template <typename T>
__global__
void add_bias_act(T* out, const T* bias, int m, int n)
{
  T val, reg_bias;

  int row_id = blockIdx.x;
  int ite = n / blockDim.x;
  int tid = threadIdx.x;

  for(int i = 0; i < ite; ++i)
  {
    reg_bias = __ldg(&bias[i * blockDim.x + tid]);
    row_id = blockIdx.x;

    while(row_id < m){
      val = out[tid + i * blockDim.x + row_id * n]+ reg_bias;
      out[tid + i * blockDim.x + row_id * n] = gelu<T>(val);
      row_id += gridDim.x;
    }
  }
}

template <>
__global__
void add_bias_act(half* out, const half* bias, int m, int n)
{
  half2 val, reg_bias;
  int row_id = blockIdx.x;
  int ite = n / blockDim.x / 2;
  int tid = threadIdx.x;

  half2* out_ptr = (half2*) out;
  const half2* bias_ptr = (half2*) bias;
  for(int i = 0; i < ite; ++i)
  {
    reg_bias = __ldg(&bias_ptr[i * blockDim.x + tid]);
    row_id = blockIdx.x;

    while(row_id < m){
      val = out_ptr[tid + i * blockDim.x + row_id * n / 2];
      val = __hadd2(val, reg_bias);
      out_ptr[tid + i * blockDim.x + row_id * n / 2] = gelu<half2>(val);
      row_id += gridDim.x;
    }
  }
}

template <typename T>
__global__
void add_bias_input_layernorm(T* out, const T* input, const T* gamma, const T* beta, int m, int n)
{
  int tid = threadIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;

  float local_out = 0.0f;
  //local_out += (float)(out[blockIdx.x * n + tid] + input[blockIdx.x * n + tid] + __ldg(&bias[tid]));
  if (tid < n)
      local_out = (float)(input[blockIdx.x * n + tid]);

  mean = blockReduceSum<float>(local_out);
  if(threadIdx.x == 0)
    s_mean = mean / n;
  __syncthreads();

  float tmp = 0.0f;
  if (tid < n)
      tmp = (local_out - s_mean) * (local_out - s_mean);
  variance = blockReduceSum<float>(tmp);
  // from TF: variance_epsilon = 1e-12 if dtype != dtypes.float16 else 1e-3
  if(threadIdx.x == 0)
    s_variance = variance / n + 1e-12f;
  __syncthreads();

  if (tid < n)
      out[blockIdx.x * n + tid] =
          (T)(((local_out - s_mean) * rsqrtf(s_variance)) * (float)(__ldg(&gamma[tid])) + (float)(__ldg(&beta[tid])));
}

// a slower version, which can have correct result
template <>
__global__
void add_bias_input_layernorm(half* out, const half* input, const half* gamma,
        const half* beta, int m, int n)
{
  int tid = threadIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;

  float local_out = 0.0f;
  //local_out += (float)(out[blockIdx.x * n + tid] + input[blockIdx.x * n + tid] + __ldg(&bias[tid]));
  if (tid < n)
      local_out = __half2float(input[blockIdx.x * n + tid]);

  mean = blockReduceSum<float>(local_out);
  if(threadIdx.x == 0)
    s_mean = mean / n;
  __syncthreads();

  float tmp = 0.0f;
  if (tid < n)
      tmp = (local_out - s_mean) * (local_out - s_mean);
  variance = blockReduceSum<float>(tmp);
  // from TF: variance_epsilon = 1e-12 if dtype != dtypes.float16 else 1e-3
  if(threadIdx.x == 0)
    s_variance = variance / n + 1e-12f;
  __syncthreads();

  if (tid < n)
  {
      float g = __half2float(gamma[tid]);
      float b = __half2float(beta[tid]);
      float tmp = (local_out - s_mean) * rsqrtf(s_variance) * g +  b;
      out[blockIdx.x * n + tid] = __float2half(tmp);
  }
}

template <typename T>
__global__
void add_bias_input_layernorm_le32(T* out, const T* input, const T* gamma, const T* beta, int m, int n)
{
  int wid = threadIdx.x >> 5;
  int lane_id = threadIdx.x & 0x1f;
  int g_wid = wid + blockIdx.x * 16;

  __shared__ float s_mean[16];
  __shared__ float s_variance[16];
  float mean =  0.0f;
  float variance = 0.0f;

  float local_out = 0.0f;
  int data_offset = blockIdx.x * 16 * n + wid * n + lane_id;
  //local_out += (float)(out[blockIdx.x * n + tid] + input[blockIdx.x * n + tid] + __ldg(&bias[tid]));
  if (lane_id < n && g_wid < m)
      local_out = (float)(input[data_offset]);

  mean = warpReduceSum<float>(local_out);
  if(lane_id == 0 && g_wid < m)
    s_mean[wid] = mean / n;
  __syncthreads();

  float tmp = 0.0f;
  if (lane_id < n && g_wid < m)
      tmp = (local_out - s_mean[wid]) * (local_out - s_mean[wid]);
  variance = warpReduceSum<float>(tmp);
  // from TF: variance_epsilon = 1e-12 if dtype != dtypes.float16 else 1e-3
  if(lane_id == 0 && g_wid < m)
    s_variance[wid] = variance / n + 1e-12f;
  __syncthreads();

  if (lane_id < n && g_wid < m)
      out[data_offset] =
          (T)(((local_out - s_mean[wid]) * rsqrtf(s_variance[wid])) * (float)(__ldg(&gamma[lane_id])) + (float)(__ldg(&beta[lane_id])));
}

// a slower version, which can have correct result
template <>
__global__
void add_bias_input_layernorm_le32(half* out, const half* input, const half* gamma,
        const half* beta, int m, int n)
{
    int wid = threadIdx.x >> 5; // warp id
    int lane_id = threadIdx.x & 0x1f;
    int g_wid = wid + blockIdx.x * 16;

    __shared__ float s_mean[16];
    __shared__ float s_variance[16];
    float mean =  0.0f;
    float variance = 0.0f;

    float local_out = 0.0f;
    int data_offset = blockIdx.x * 16 * n + wid * n + lane_id;

    if (lane_id < n && g_wid < m)
        local_out = __half2float(input[data_offset]);

    mean = warpReduceSum<float>(local_out);
    if(lane_id == 0 && g_wid < m)
        s_mean[wid] = mean / n;
    __syncthreads();

    float tmp = 0.0f;
    if (lane_id < n && g_wid < m)
        tmp = (local_out - s_mean[wid]) * (local_out - s_mean[wid]);

    variance = warpReduceSum<float>(tmp);
    // from TF: variance_epsilon = 1e-12 if dtype != dtypes.float16 else 1e-3
    if(lane_id == 0)
        s_variance[wid] = variance / n + 1e-12f;
    __syncthreads();

    if (lane_id < n && g_wid < m)
    {
        float g = __half2float(gamma[lane_id]);
        float b = __half2float(beta[lane_id]);
        float tmp = (local_out - s_mean[wid]) * rsqrtf(s_variance[wid]) * g +  b;
        out[data_offset] = __float2half(tmp);
    }
}

template <typename T>
__global__
void add_bias_input_layernorm_v2(T* out, const T* __restrict input,
                                const T* __restrict gamma, const T* __restrict beta, int n)
{
  const int ite = 4;
  int tid = threadIdx.x;
  const int bid = blockIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;
  float local_out[ite];

  float sum = 0.0f;
  #pragma unroll
  for(int i = 0; i < ite; i++)
  {
    int col_id = i * blockDim.x + tid;
    int id = bid * n + col_id;
    local_out[i] = (float)(__ldg(&input[id]));
    sum += local_out[i];
  }

  mean = blockReduceSum<float>(sum);
  if(tid == 0)
    s_mean = mean / n;
  __syncthreads();

  float var = 0.0f;
  #pragma unroll
  for(int i = 0; i < ite; i++)
  {
    float diff = local_out[i] - s_mean;
    var += diff * diff;
  }

  variance = blockReduceSum<float>(var);
  if(tid == 0)
    s_variance = rsqrtf(variance / n + 1e-12f);
  __syncthreads();

  #pragma unroll
  for(int i = 0; i < ite; i++)
  {
    int col_id = i * blockDim.x + tid;
    int id = bid * n + col_id;
    out[id] = (T)((local_out[i] - s_mean) * s_variance * (float)__ldg(&gamma[col_id]) + (float)__ldg(&beta[col_id]));
  }
}

template <>
__global__
void add_bias_input_layernorm_v2(half* out, const half* __restrict input,
  const half* __restrict gamma, const half* __restrict beta, int n)
{
  const int ite = 4;
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean =  0.0f;
  float variance = 0.0f;
  half2 local_out_half2[ite];

  half2* out_ptr = (half2*)out;
  const half2* input_ptr = (const half2*)input;
  //const half2* bias_ptr = (const half2*)bias;
  const half2* gamma_ptr = (const half2*)gamma;
  const half2* beta_ptr = (const half2*)beta;

  // float sum = 0.0f;
  half2 sum = __float2half2_rn(0.0f);
  #pragma unroll
  for(int i = 0; i < ite; i++)
  {
    int col_id = i * blockDim.x + tid;
    int id = bid * n / 2 + col_id;
    local_out_half2[i] = __ldg(&input_ptr[id]);
    sum += local_out_half2[i];
  }

  mean = blockReduceSum<float>((float)(sum.x + sum.y));
  if(threadIdx.x == 0)
    s_mean = mean / n;
  __syncthreads();

  float var = 0.0f;
  half2 s_mean_2 = __float2half2_rn(s_mean);
  #pragma unroll
  for(int i = 0; i < ite; i++)
  {
    local_out_half2[i] = local_out_half2[i] - s_mean_2;
    float v1 = (float)local_out_half2[i].x;
    float v2 = (float)local_out_half2[i].y;
    var += v1 * v1 + v2 * v2;
  }

  variance = blockReduceSum<float>(var);
  if(threadIdx.x == 0)
    s_variance = rsqrtf(variance / n + 1e-12f);
  __syncthreads();

  half2 s_var_2 = __float2half2_rn(s_variance);
  #pragma unroll
  for(int i = 0; i < ite; i++)
  {
    int col_id = i * blockDim.x + tid;
    int id = bid * n / 2 + col_id;
    out_ptr[id] = local_out_half2[i] * s_var_2 * __ldg(&gamma_ptr[col_id]) + __ldg(&beta_ptr[col_id]);
  }
}

template <typename T>
void add_bias_act_kernelLauncher(T* out, const T* bias, int m, int n, cudaStream_t stream)
{
  dim3 grid(ceil(m / 4.));
  dim3 block(n / 4);
  add_bias_act<T><<<grid, block, 0, stream>>>(out, bias, m, n);
}

template<typename T>
void layernorm_kernelLauncher(T* out, const T* input,
  const T* gamma, const T* beta, int m, int n, cudaStream_t stream)
{
  dim3 grid(m);
  dim3 block(n);
  if(n == 768 || n == 1024)
    add_bias_input_layernorm_v2<T><<<grid, n / 4, 0, stream>>>(out, input, gamma, beta, n);
  else if (n > 32)
    add_bias_input_layernorm<T><<<grid, block, 0, stream>>>(out, input, gamma, beta, m, n);
  else {
      grid.x = (m + 15) / 16;
      block.x = 512;
    add_bias_input_layernorm_le32<T><<<grid, block, 0, stream>>>(out, input, gamma, beta, m, n);
  }
}

template <>
void layernorm_kernelLauncher(half* out, const half* input,
  const half* gamma, const half* beta, int m, int n, cudaStream_t stream)
{
    dim3 grid(m);
    dim3 block(n);

    if(m >= 512 && (n == 768 || n == 1024))
        add_bias_input_layernorm_v2<half><<<grid, n / 8, 0, stream>>>((half*)out, (half*)input, (half*)gamma, (half*)beta, n);
    else if (n>32)
        add_bias_input_layernorm<half><<<grid, n, 0, stream>>>((half*)out, (half*)input, (half*)gamma, (half*)beta, m, n);
    else
    {
        grid.x = (m + 15) / 16;
        block.x = 512; // 32 * 16
        add_bias_input_layernorm_le32<half><<<grid, block, 0, stream>>>((half*)out, (half*)input, (half*)gamma, (half*)beta, m, n);
  }
}


namespace cuda{

/**
* Multi-head attetion open sourced
*/
  __inline__ __device__
int target_index(int id1, int id2, int id3, int id4, int dim_1, int dim_2, int dim_3, int dim_4)
{
  return id1 * (dim_2 * dim_3 * dim_4) + id3 * (dim_2 * dim_4) + id2 * dim_4 + id4;
}

  __inline__ __device__
int target_index_taobao(int id1, int id2, int id3, int id4, int dim_1, int dim_2, int dim_3, int dim_4)
{
  return id3 * (dim_2 * dim_1 * dim_4) + id1 * (dim_2 * dim_4) + id2 * dim_4 + id4;
}

__inline__ __device__
int get_qkv_id(int batch_size, int word_per_block,
        int m0, int m1)
{
    int bid = blockIdx.x * word_per_block;

    if (bid < m0) {
        return 0;
    } else if (bid < m0 + m1) {
        return 1;
    } else {
        return 2;
    }
}

__inline__ __device__
int get_batch_id(int qkv_id, int word_per_block,
        int from_seq_len, int to_seq_len, int batch_size)
{
    if (qkv_id == 0)
    {
        return (blockIdx.x * word_per_block) / from_seq_len;
    } else if(qkv_id == 1)
    {
        int tmp = batch_size * from_seq_len;
        return (blockIdx.x * word_per_block - tmp) / to_seq_len;
    } else {
        int tmp = batch_size * (from_seq_len + to_seq_len);
        return (blockIdx.x * word_per_block - tmp) / to_seq_len;
    }
}

__inline__ __device__
int get_word_id(int qkv_id, int word_per_block,
        int from_seq_len, int to_seq_len, int batch_size)
{
    if (qkv_id == 0)
    {
        return (blockIdx.x * word_per_block) % from_seq_len;
    } else if(qkv_id == 1)
    {
        int tmp = batch_size * from_seq_len;
        return (blockIdx.x * word_per_block - tmp) % to_seq_len;
    } else {
        int tmp = batch_size * (from_seq_len + to_seq_len);
        return (blockIdx.x * word_per_block - tmp) % to_seq_len;
    }
}

template <typename T>
__inline__ __device__
T my_leaky_relu(T x, float alpha) {return x;}

template <>
__inline__ __device__
float my_leaky_relu(float x, float alpha) {
    if (x < 0)
        return x * alpha;
    else
        return x;
}

template <>
__inline__ __device__
half2 my_leaky_relu(half2 x, float alpha)
{
    half h_alpha = __float2half(alpha);
    half h_zero = __float2half(0.0f);
    half2 tmp;
    if (__hgt(x.x, h_zero) == true)
        tmp.x = x.x;
    else
        tmp.x = __hmul(x.x, h_alpha);

    if (__hgt(x.y, h_zero) == true)
        tmp.y = x.y;
    else
        tmp.y = __hmul(x.y, h_alpha);

    return tmp;
}

template <>
__inline__ __device__
half my_leaky_relu(half x, float alpha)
{
    half h_alpha = __float2half(alpha);
    half h_zero = __float2half(0.0f);
    if (__hgt(x, h_zero) == true)
        return x;
    else
        return __hmul(x, h_alpha);
}

template<typename T>
__global__
void add_QKV_bias(T* Q, const T* bias_Q, T* K, const T* bias_K, T* V, const T* bias_V, T* q_buf_, T* k_buf_, T* v_buf_,
  const int batch_size, const int from_seq_len, const int to_seq_len, const int head_num, const int size_per_head,
  bool do_lrelu, float lrelu_alpha)
{
  T* data_ptr;
  T* buf_ptr;
  const T* bias_ptr;

  int m0 = batch_size * from_seq_len;
  int m1 = batch_size * to_seq_len;
  int n = head_num * size_per_head;

  int qkv_id = get_qkv_id(batch_size, 1, m0, m1);
  //int row_offset = (blockIdx.x * 1 % m) * n;
  int batch_id = get_batch_id(qkv_id, 1, from_seq_len, to_seq_len, batch_size);
  int head_id = threadIdx.x / size_per_head;
  int id_in_head = threadIdx.x % size_per_head;
  // seq_id
  int word_start_id = get_word_id(qkv_id, 1, from_seq_len, to_seq_len, batch_size);
  int gather_offset = (batch_id * to_seq_len + word_start_id) * n;
  int seq_len = 0;

  if(qkv_id == 0)
  {
    data_ptr = Q + blockIdx.x * 1 % m0 * n;
    buf_ptr = q_buf_;
    bias_ptr = bias_Q;
    seq_len = from_seq_len;
  }
  else if(qkv_id == 1)
  {
    data_ptr = K + gather_offset;
    //data_ptr = K + row_offset;
    buf_ptr = k_buf_;
    bias_ptr = bias_K;
    seq_len = to_seq_len;
  }
  else
  {
    data_ptr = V + gather_offset;
    //data_ptr = V + row_offset;
    buf_ptr = v_buf_;
    bias_ptr = bias_V;
    seq_len = to_seq_len;
  }

  T bias = __ldg(&bias_ptr[threadIdx.x]);

  for(int i = word_start_id; i < word_start_id + 1; ++i)
  {
    T tmp = data_ptr[threadIdx.x] + bias;

    //// transpose (batch_size, num_heads, seq_len, size_per_head)
    //int target_id = batch_id * (seq_len * head_num * size_per_head) + head_id * seq_len * size_per_head + 
    //  i * size_per_head + id_in_head;
    // transpose (num_heads, batch_size, seq_len, size_per_head)
    int target_id = head_id * (seq_len * batch_size * size_per_head) + batch_id * seq_len * size_per_head + 
      i * size_per_head + id_in_head;

    if (do_lrelu)
    {
        tmp = my_leaky_relu<T>(tmp, lrelu_alpha);
    }
    buf_ptr[target_id] = tmp;
    data_ptr += n;
  }
}

template<typename T>
__global__
void add_QKV_bias_qk(T* Q, const T* bias_Q, T* K, const T* bias_K, T* q_buf_, T* k_buf_,
  const int batch_size, const int from_seq_len, const int to_seq_len, const int head_num, const int size_per_head,
  bool do_lrelu, float lrelu_alpha)
{
  T* data_ptr;
  T* buf_ptr;
  const T* bias_ptr;

  int m0 = batch_size * from_seq_len;
  int m1 = batch_size * to_seq_len;
  int n = head_num * size_per_head;

  int qkv_id = get_qkv_id(batch_size, 1, m0, m1);
  //int row_offset = (blockIdx.x * 1 % m) * n;
  int batch_id = get_batch_id(qkv_id, 1, from_seq_len, to_seq_len, batch_size);
  int head_id = threadIdx.x / size_per_head;
  int id_in_head = threadIdx.x % size_per_head;
  // seq_id
  int word_start_id = get_word_id(qkv_id, 1, from_seq_len, to_seq_len, batch_size);
  int gather_offset = (batch_id * to_seq_len + word_start_id) * n;
  int seq_len = 0;

  if(qkv_id == 0)
  {
    data_ptr = Q + blockIdx.x * 1 % m0 * n;
    buf_ptr = q_buf_;
    bias_ptr = bias_Q;
    seq_len = from_seq_len;
  }
  else if(qkv_id == 1)
  {
    data_ptr = K + gather_offset;
    //data_ptr = K + row_offset;
    buf_ptr = k_buf_;
    bias_ptr = bias_K;
    seq_len = to_seq_len;
  }

  T bias = __ldg(&bias_ptr[threadIdx.x]);

  for(int i = word_start_id; i < word_start_id + 1; ++i)
  {
    T tmp = data_ptr[threadIdx.x] + bias;

    // transpose (num_heads, batch_size, seq_len, size_per_head)
    int target_id = head_id * (seq_len * batch_size * size_per_head) + batch_id * seq_len * size_per_head + 
      i * size_per_head + id_in_head;

    if (do_lrelu)
    {
        tmp = my_leaky_relu<T>(tmp, lrelu_alpha);
    }
    buf_ptr[target_id] = tmp;
    data_ptr += n;
  }
}

template<typename T>
__global__
void add_QKV_bias_v(T* V, const T* bias_V, T* v_buf_,
  const int batch_size, const int from_seq_len, const int to_seq_len,
  const int head_num, const int size_per_head,
  bool do_lrelu, float lrelu_alpha)
{
  T* data_ptr;
  T* buf_ptr;
  const T* bias_ptr;

  int n = head_num * size_per_head;

  int batch_id = blockIdx.x / to_seq_len;
  int head_id = threadIdx.x / size_per_head;
  int id_in_head = threadIdx.x % size_per_head;
  // seq_id
  int word_start_id = (blockIdx.x * 1) % to_seq_len;
  int gather_offset = (batch_id * to_seq_len + word_start_id) * n;
  int seq_len = 0;

  data_ptr = V + gather_offset;
  //data_ptr = V + row_offset;
  buf_ptr = v_buf_;
  bias_ptr = bias_V;
  seq_len = to_seq_len;

  T bias = __ldg(&bias_ptr[threadIdx.x]);

  for(int i = word_start_id; i < word_start_id + 1; ++i)
  {
    T tmp = data_ptr[threadIdx.x] + bias;

    // transpose (num_heads, batch_size, seq_len, size_per_head)
    int target_id = head_id * (seq_len * batch_size * size_per_head) + batch_id * seq_len * size_per_head + 
      i * size_per_head + id_in_head;

    if (do_lrelu)
    {
        tmp = my_leaky_relu<T>(tmp, lrelu_alpha);
    }
    buf_ptr[target_id] = tmp;
    data_ptr += n;
  }
}

template <typename T>
__global__
void add_QKV_bias_Q(T* Q, const T* bias_Q, T* q_buf_,
  const int batch_size, const int seq_len, const int head_num, const int size_per_head,
  const int word_per_block, bool do_lrelu, float lrelu_alpha) {
  int tid = blockIdx.x * head_num * size_per_head + threadIdx.x;

  int batch_id = blockIdx.x / seq_len;
  int seq_id = blockIdx.x % seq_len;
  int head_id = threadIdx.x / size_per_head;
  int id = threadIdx.x % size_per_head;
  int target_id = target_index(head_id, seq_id, batch_id, id, head_num, seq_len, batch_size, size_per_head);

  int bias_id = threadIdx.x;

  float* src_ptr = (float*)Q;
  float* dst_ptr = (float*)q_buf_;
  const float* bias_ptr = (const float*)bias_Q;
  float tmp = 0.0f;
  if (bias_id < head_num * size_per_head) {
      tmp = src_ptr[tid] + bias_ptr[bias_id];

  }
  if (do_lrelu)
  {
      tmp = my_leaky_relu<float>(tmp, lrelu_alpha);
  }

  if (bias_id < head_num * size_per_head)
      dst_ptr[target_id] = tmp;
}

template <>
__global__
void add_QKV_bias_Q(half* Q, const half* bias_Q, half* q_buf_, const int batch_size, const int seq_len,
        const int head_num, const int size_per_head, const int word_per_block, bool do_lrelu, float lrelu_alpha)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int batch_id = tid / (head_num * seq_len * size_per_head);
  int seq_id = (tid % (head_num * seq_len * size_per_head)) / (head_num * size_per_head);
  int head_id = (tid % (head_num * size_per_head)) / size_per_head;
  int id = tid % size_per_head;
  int target_id = target_index(head_id, seq_id, batch_id, id, head_num, seq_len, batch_size, size_per_head);

  int bias_id = threadIdx.x;

  half2* src_ptr = (half2*)Q;
  half2* dst_ptr = (half2*)q_buf_;
  const half2* bias_ptr = (const half2*)bias_Q;
  half2 tmp = __hadd2(src_ptr[tid],  __ldg(&bias_ptr[bias_id]));
  if (do_lrelu)
  {
      tmp = my_leaky_relu<half2>(tmp, lrelu_alpha);
  }
  dst_ptr[target_id] = tmp;
}

template <typename T>
__global__
void add_QKV_bias_KV(T* K, const T* bias_K, T* V, const T* bias_V, T* k_buf_, T* v_buf_,
  const int batch_size, const int seq_len, const int head_num, const int size_per_head, const int word_per_block,
  bool do_lrelu, float lrelu_alpha) {}

template <>
__global__
void add_QKV_bias_KV(half* K, const half* bias_K, half* V, const half* bias_V, half* k_buf_, half* v_buf_, const int batch_size,
        const int seq_len, const int head_num, const int size_per_head, const int word_per_block, bool do_lrelu, float lrelu_alpha)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int batch_id = tid / (head_num * seq_len * size_per_head);
  int seq_id = (tid % (head_num * seq_len * size_per_head)) / (head_num * size_per_head);
  int head_id = (tid % (head_num * size_per_head)) / size_per_head;
  int id = tid % size_per_head;
  int target_id = target_index(head_id, seq_id, batch_id, id, head_num, seq_len, batch_size, size_per_head);
  //int target_id = target_index(batch_id, seq_id, head_id, id, batch_size, seq_len, head_num, size_per_head);
  int qv_tid = batch_id * seq_len * blockDim.x +
               seq_id * blockDim.x + threadIdx.x;

  int bias_id = threadIdx.x;

  half2* src_ptr = (half2*)K;
  half2* dst_ptr = (half2*)k_buf_;
  const half2* bias_ptr = (const half2*)bias_K;
  half2 tmp;
  tmp = __hadd2(src_ptr[qv_tid],  __ldg(&bias_ptr[bias_id]));
  if (do_lrelu)
  {
      tmp = my_leaky_relu<half2>(tmp, lrelu_alpha);
  }
  dst_ptr[target_id] = tmp;

  src_ptr = (half2*)V;
  dst_ptr = (half2*)v_buf_;
  bias_ptr = (const half2*)bias_V;
  tmp = __hadd2(src_ptr[qv_tid],  __ldg(&bias_ptr[bias_id]));
  if (do_lrelu)
  {
      tmp = my_leaky_relu<half2>(tmp, lrelu_alpha);
  }
  dst_ptr[target_id] = tmp;
}

__global__
void add_QKV_bias_KV_one(half* K_V, const half* bias_K_V, half* k_v_buf_, const int batch_size,
        const int seq_len, const int head_num, const int size_per_head, const int word_per_block, bool do_lrelu, float lrelu_alpha)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int batch_id = tid / (head_num * seq_len * size_per_head);
  int seq_id = (tid % (head_num * seq_len * size_per_head)) / (head_num * size_per_head);
  int head_id = (tid % (head_num * size_per_head)) / size_per_head;
  int id = tid % size_per_head;
  int target_id = target_index(head_id, seq_id, batch_id, id, head_num, seq_len, batch_size, size_per_head);
  //int target_id = target_index(batch_id, seq_id, head_id, id, batch_size, seq_len, head_num, size_per_head);
  int qv_tid = batch_id * seq_len * blockDim.x +
               seq_id * blockDim.x + threadIdx.x;

  int bias_id = threadIdx.x;

  half2* src_ptr = (half2*)K_V;
  half2* dst_ptr = (half2*)k_v_buf_;
  const half2* bias_ptr = (const half2*)bias_K_V;
  half2 tmp;
  tmp = __hadd2(src_ptr[qv_tid],  __ldg(&bias_ptr[bias_id]));
  if (do_lrelu)
  {
      tmp = my_leaky_relu<half2>(tmp, lrelu_alpha);
  }
  dst_ptr[target_id] = tmp;
}


template <typename T>
__global__
void add_QKV_bias_KV_one_m1(T* K_V, const T* bias_K_V, T* k_v_buf_,const int batch_size,
        const int seq_len, const int head_num, const int size_per_head, const int word_per_block, bool do_lrelu, float lrelu_alpha)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  //int batch_id = blockIdx.x / (head_num * seq_len);
  int seq_id = (blockIdx.x / head_num) % seq_len;
  int head_id = blockIdx.x % head_num;
  //int id = tid % size_per_head;
  int target_id = threadIdx.x + seq_id * size_per_head + head_id * seq_len * size_per_head;

  int bias_id = threadIdx.x + head_id * size_per_head;

  float* src_ptr = (float*)K_V;
  float* dst_ptr = (float*)k_v_buf_;
  const float* bias_ptr = (const float*)bias_K_V;
  float tmp = src_ptr[tid] + bias_ptr[bias_id];;
  if (do_lrelu)
  {
      tmp = my_leaky_relu<float>(tmp, lrelu_alpha);
  }

  dst_ptr[target_id] = tmp;
}

template <>
__global__
void add_QKV_bias_KV_one_m1(half* K_V, const half* bias_K_V, half* k_v_buf_,const int batch_size,
        const int seq_len, const int head_num, const int size_per_head, const int word_per_block, bool do_lrelu, float lrelu_alpha)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  //int batch_id = blockIdx.x / (head_num * seq_len);
  int seq_id = (blockIdx.x / head_num) % seq_len;
  int head_id = blockIdx.x % head_num;
  //int id = tid % size_per_head;
  int target_id = threadIdx.x + seq_id * size_per_head + head_id * seq_len * size_per_head;

  int bias_id = threadIdx.x + head_id * size_per_head;

  half2* src_ptr = (half2*)K_V;
  half2* dst_ptr = (half2*)k_v_buf_;
  const half2* bias_ptr = (const half2*)bias_K_V;
  half2 tmp;
  tmp = __hadd2(src_ptr[tid],  __ldg(&bias_ptr[bias_id]));
  if (do_lrelu)
  {
      tmp = my_leaky_relu<half2>(tmp, lrelu_alpha);
  }

  dst_ptr[target_id] = tmp;
}

template <typename T>
__global__
void add_QKV_bias_KV_m1(T* K, const T* bias_K, T* V, const T* bias_V, T* k_buf_, T* v_buf_, const int batch_size,
        const int seq_len, const int head_num, const int size_per_head, const int word_per_block, bool do_lrelu, float lrelu_alpha)
{
  int tid = blockIdx.x * size_per_head + threadIdx.x;
  int seq_id = (blockIdx.x / head_num) % seq_len;
  int head_id = blockIdx.x % head_num;
  int target_id = threadIdx.x + seq_id * size_per_head + head_id * seq_len * size_per_head;

  int bias_id = threadIdx.x + head_id * size_per_head;

  float* src_ptr = (float*)K;
  float* dst_ptr = (float*)k_buf_;
  const float* bias_ptr = (const float*)bias_K;
  float tmp = 0.0f;
  if (threadIdx.x < size_per_head)
      tmp = src_ptr[tid] + bias_ptr[bias_id];

  if (do_lrelu)
  {
      tmp = my_leaky_relu<float>(tmp, lrelu_alpha);
  }

  if (threadIdx.x < size_per_head)
      dst_ptr[target_id] = tmp;

  src_ptr = (float*)V;
  dst_ptr = (float*)v_buf_;
  bias_ptr = (const float*)bias_V;
  if (threadIdx.x < size_per_head)
      tmp = src_ptr[tid] + bias_ptr[bias_id];
  if (do_lrelu)
  {
      tmp = my_leaky_relu<float>(tmp, lrelu_alpha);
  }
  if (threadIdx.x < size_per_head)
      dst_ptr[target_id] = tmp;
}

__global__
void add_QKV_bias_KV_m1(half* K, const half* bias_K, half* V, const half* bias_V, half* k_buf_, half* v_buf_, const int batch_size,
        const int seq_len, const int head_num, const int size_per_head, const int word_per_block, bool do_lrelu, float lrelu_alpha)
{
  int tid = blockIdx.x * size_per_head + threadIdx.x;
  int seq_id = (blockIdx.x / head_num) % seq_len;
  int head_id = blockIdx.x % head_num;
  int target_id = threadIdx.x + seq_id * size_per_head + head_id * seq_len * size_per_head;

  int bias_id = threadIdx.x + head_id * size_per_head;

  half2* src_ptr = (half2*)K;
  half2* dst_ptr = (half2*)k_buf_;
  const half2* bias_ptr = (const half2*)bias_K;
  half2 tmp;
  tmp.x = __float2half(0.0f);
  tmp.y = __float2half(0.0f);
  if (threadIdx.x < size_per_head)
      tmp = __hadd2(src_ptr[tid],  __ldg(&bias_ptr[bias_id]));

  if (do_lrelu)
  {
      tmp = my_leaky_relu<half2>(tmp, lrelu_alpha);
  }

  if (threadIdx.x < size_per_head)
      dst_ptr[target_id] = tmp;

  src_ptr = (half2*)V;
  dst_ptr = (half2*)v_buf_;
  bias_ptr = (const half2*)bias_V;
  if (threadIdx.x < size_per_head)
      tmp = __hadd2(src_ptr[tid],  __ldg(&bias_ptr[bias_id]));
  if (do_lrelu)
  {
      tmp = my_leaky_relu<half2>(tmp, lrelu_alpha);
  }
  if (threadIdx.x < size_per_head)
      dst_ptr[target_id] = tmp;
}


template <typename T>
__global__
void add_QKV_bias_ln_Q(T* Q, const T* bias_Q, T* q_buf_,
  const int batch_size, const int seq_len, const int head_num, const int size_per_head,
  const int word_per_block, bool do_lrelu, float lrelu_alpha)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int batch_id = tid / (head_num * seq_len * size_per_head);
  int seq_id = (tid % (head_num * seq_len * size_per_head)) / (head_num * size_per_head);
  int head_id = (tid % (head_num * size_per_head)) / size_per_head;
  int id = tid % size_per_head;
  int target_id = target_index(head_id, seq_id, batch_id, id, head_num, seq_len, batch_size, size_per_head);

  int bias_id = threadIdx.x;

  float* src_ptr = (float*)Q;
  float* dst_ptr = (float*)q_buf_;
  const float* bias_ptr = (const float*)bias_Q;
  float tmp = src_ptr[tid] + bias_ptr[bias_id];

  if (do_lrelu)
  {
      tmp = my_leaky_relu<float>(tmp, lrelu_alpha);
  }
  dst_ptr[target_id] = tmp;
}

template <>
__global__
void add_QKV_bias_ln_Q(half* Q, const half* bias_Q, half* q_buf_, const int batch_size, const int seq_len,
        const int head_num, const int size_per_head, const int word_per_block, bool do_lrelu, float lrelu_alpha)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int batch_id = tid / (head_num * seq_len * size_per_head);
  int seq_id = (tid % (head_num * seq_len * size_per_head)) / (head_num * size_per_head);
  int head_id = (tid % (head_num * size_per_head)) / size_per_head;
  int id = tid % size_per_head;
  int target_id = target_index(head_id, seq_id, batch_id, id, head_num, seq_len, batch_size, size_per_head);

  int bias_id = threadIdx.x;

  half2* src_ptr = (half2*)Q;
  half2* dst_ptr = (half2*)q_buf_;
  const half2* bias_ptr = (const half2*)bias_Q;
  half2 tmp = __hadd2(src_ptr[tid],  __ldg(&bias_ptr[bias_id]));
  if (do_lrelu)
  {
      tmp = my_leaky_relu<half2>(tmp, lrelu_alpha);
  }

  dst_ptr[target_id] = tmp;
}

template <typename T>
__global__
void add_QKV_bias_ln_KV(T* K, const T* bias_K, T* V, const T* bias_V, T* k_buf_, T* v_buf_, 
  const int batch_size, const int seq_len, const int head_num, const int size_per_head, const int word_per_block,
  bool do_lrelu, float lrelu_alpha) {}

template <>
__global__
void add_QKV_bias_ln_KV(half* K, const half* bias_K, half* V, const half* bias_V, half* k_buf_, half* v_buf_, const int batch_size,
        const int seq_len, const int head_num, const int size_per_head, const int word_per_block, bool do_lrelu, float lrelu_alpha)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int batch_id = blockIdx.x / seq_len;
  int seq_id = blockIdx.x % seq_len;
  int head_id =  threadIdx.x / size_per_head;
  int id = tid % size_per_head;
  int target_id = target_index(head_id, seq_id, batch_id, id, head_num, seq_len, batch_size, size_per_head);
  //int target_id = target_index(batch_id, seq_id, head_id, id, batch_size, seq_len, head_num, size_per_head);
  int qv_tid = batch_id * seq_len * blockDim.x +
               seq_id * blockDim.x + threadIdx.x;

  int bias_id = threadIdx.x;

  half2* src_ptr = (half2*)K;
  half2* dst_ptr = (half2*)k_buf_;
  const half2* bias_ptr = (const half2*)bias_K;
  half2 tmp;
  tmp = __hadd2(src_ptr[qv_tid],  __ldg(&bias_ptr[bias_id]));
  if (do_lrelu)
  {
      tmp = my_leaky_relu<half2>(tmp, lrelu_alpha);
  }

  dst_ptr[target_id] = tmp;

  src_ptr = (half2*)V;
  dst_ptr = (half2*)v_buf_;
  bias_ptr = (const half2*)bias_V;
  tmp = __hadd2(src_ptr[qv_tid],  __ldg(&bias_ptr[bias_id]));
  if (do_lrelu)
  {
      tmp = my_leaky_relu<half2>(tmp, lrelu_alpha);
  }
  dst_ptr[target_id] = tmp;
}

__global__
void add_QKV_bias_ln_KV_K(half* K, const half* bias_K, half* k_buf_, const int batch_size,
        const int seq_len, const int head_num, const int size_per_head, const int word_per_block, bool do_lrelu, float lrelu_alpha)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int batch_id = blockIdx.x / seq_len;
  int seq_id = blockIdx.x % seq_len;
  int head_id =  threadIdx.x / size_per_head;
  int id = tid % size_per_head;
  int target_id = target_index(head_id, seq_id, batch_id, id, head_num, seq_len, batch_size, size_per_head);
  //int target_id = target_index(batch_id, seq_id, head_id, id, batch_size, seq_len, head_num, size_per_head);
  int qv_tid = batch_id * seq_len * blockDim.x +
               seq_id * blockDim.x + threadIdx.x;

  int bias_id = threadIdx.x;

  half2* src_ptr = (half2*)K;
  half2* dst_ptr = (half2*)k_buf_;
  const half2* bias_ptr = (const half2*)bias_K;
  half2 tmp;
  tmp = __hadd2(src_ptr[qv_tid],  __ldg(&bias_ptr[bias_id]));
  if (do_lrelu)
  {
      tmp = my_leaky_relu<half2>(tmp, lrelu_alpha);
  }

  dst_ptr[target_id] = tmp;
}

__global__
void add_QKV_bias_ln_KV_V(half* V, const half* bias_V, half* v_buf_, const int batch_size,
        const int seq_len, const int head_num, const int size_per_head, const int word_per_block, bool do_lrelu, float lrelu_alpha)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int batch_id = blockIdx.x / seq_len;
  int seq_id = blockIdx.x % seq_len;
  int head_id =  threadIdx.x / size_per_head;
  int id = tid % size_per_head;
  int target_id = target_index(head_id, seq_id, batch_id, id, head_num, seq_len, batch_size, size_per_head);
  //int target_id = target_index(batch_id, seq_id, head_id, id, batch_size, seq_len, head_num, size_per_head);
  int qv_tid = batch_id * seq_len * blockDim.x +
               seq_id * blockDim.x + threadIdx.x;

  int bias_id = threadIdx.x;

  half2 tmp;
  half2* src_ptr = (half2*)V;
  half2* dst_ptr = (half2*)v_buf_;
  const half2* bias_ptr = (const half2*)bias_V;
  tmp = __hadd2(src_ptr[qv_tid],  __ldg(&bias_ptr[bias_id]));
  if (do_lrelu)
  {
      tmp = my_leaky_relu<half2>(tmp, lrelu_alpha);
  }
  dst_ptr[target_id] = tmp;
}

template <typename T>
__global__
void add_QKV_bias_ln_KV_m1(T* K, const T* bias_K, T* V, const T* bias_V, T* k_buf_, T* v_buf_, const int batch_size,
        const int seq_len, const int head_num, const int size_per_head, const int word_per_block, bool do_lrelu, float lrelu_alpha)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int seq_id = blockIdx.x % seq_len;
  int head_id =  threadIdx.x / size_per_head;
  int id = tid % size_per_head;
  int bias_id = threadIdx.x;

  float* src_ptr = (float*)K;
  float* dst_ptr = (float*)k_buf_;
  const float* bias_ptr = (const float*)bias_K;
  float tmp = src_ptr[tid] + bias_ptr[bias_id];
  if (do_lrelu)
  {
      tmp = my_leaky_relu<float>(tmp, lrelu_alpha);
  }

  //for (int i = 0; i < batch_size; i++)
  {
      //int target_id = target_index(i, seq_id, head_id, id, batch_size, seq_len, head_num, size_per_head);
      int target_id =  id + seq_id * size_per_head + head_id * size_per_head * seq_len;
      dst_ptr[target_id] = tmp;
  }

  float* src_ptr_v = (float*)V;
  float* dst_ptr_v = (float*)v_buf_;
  bias_ptr = (const float*)bias_V;
  tmp = src_ptr_v[tid] + bias_ptr[bias_id];
  if (do_lrelu)
  {
      tmp = my_leaky_relu<float>(tmp, lrelu_alpha);
  }
  //for (int i = 0; i < batch_size; i++)
  {
      //int target_id = target_index(i, seq_id, head_id, id, batch_size, seq_len, head_num, size_per_head);
      int target_id =  id + seq_id * size_per_head + head_id * size_per_head * seq_len;
      dst_ptr_v[target_id] = tmp;
  }
}

template <>
__global__
void add_QKV_bias_ln_KV_m1(half* K, const half* bias_K, half* V, const half* bias_V, half* k_buf_, half* v_buf_, const int batch_size,
        const int seq_len, const int head_num, const int size_per_head, const int word_per_block, bool do_lrelu, float lrelu_alpha)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  //int batch_id = blockIdx.x / seq_len;
  int seq_id = blockIdx.x % seq_len;
  int head_id =  threadIdx.x / size_per_head;
  int id = tid % size_per_head;
  //int target_id = target_index(batch_id, seq_id, head_id, id, batch_size, seq_len, head_num, size_per_head);
  int bias_id = threadIdx.x;

  half2* src_ptr = (half2*)K;
  half2* dst_ptr = (half2*)k_buf_;
  const half2* bias_ptr = (const half2*)bias_K;
  half2 tmp;
  tmp = __hadd2(src_ptr[tid],  __ldg(&bias_ptr[bias_id]));
  if (do_lrelu)
  {
      tmp = my_leaky_relu<half2>(tmp, lrelu_alpha);
  }

  //for (int i = 0; i < batch_size; i++)
  {
      //int target_id = target_index(i, seq_id, head_id, id, batch_size, seq_len, head_num, size_per_head);
      int target_id =  id + seq_id * size_per_head + head_id * size_per_head * seq_len;
      dst_ptr[target_id] = tmp;
  }

  half2* src_ptr_v = (half2*)V;
  half2* dst_ptr_v = (half2*)v_buf_;
  bias_ptr = (const half2*)bias_V;
  tmp = __hadd2(src_ptr_v[tid],  __ldg(&bias_ptr[bias_id]));
  if (do_lrelu)
  {
      tmp = my_leaky_relu<half2>(tmp, lrelu_alpha);
  }
  //for (int i = 0; i < batch_size; i++)
  {
      //int target_id = target_index(i, seq_id, head_id, id, batch_size, seq_len, head_num, size_per_head);
      int target_id =  id + seq_id * size_per_head + head_id * size_per_head * seq_len;
      dst_ptr_v[target_id] = tmp;
  }
}

template <typename T>
__global__
void add_QKV_bias_ln_KV_m1_K(T* K, const T* bias_K, T* k_buf_, const int batch_size,
        const int seq_len, const int head_num, const int size_per_head, const int word_per_block, bool do_lrelu, float lrelu_alpha)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int seq_id = blockIdx.x % seq_len;
  int head_id =  threadIdx.x / size_per_head;
  int id = tid % size_per_head;
  int bias_id = threadIdx.x;

  float* src_ptr = (float*)K;
  float* dst_ptr = (float*)k_buf_;
  const float* bias_ptr = (const float*)bias_K;
  float tmp = src_ptr[tid] + bias_ptr[bias_id];
  if (do_lrelu)
  {
      tmp = my_leaky_relu<float>(tmp, lrelu_alpha);
  }

  //for (int i = 0; i < batch_size; i++)
  {
      //int target_id = target_index(i, seq_id, head_id, id, batch_size, seq_len, head_num, size_per_head);
      int target_id =  id + seq_id * size_per_head + head_id * size_per_head * seq_len;
      dst_ptr[target_id] = tmp;
  }
}

template <>
__global__
void add_QKV_bias_ln_KV_m1_K(half* K, const half* bias_K, half* k_buf_, const int batch_size,
        const int seq_len, const int head_num, const int size_per_head, const int word_per_block, bool do_lrelu, float lrelu_alpha)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  //int batch_id = blockIdx.x / seq_len;
  int seq_id = blockIdx.x % seq_len;
  int head_id =  threadIdx.x / size_per_head;
  int id = tid % size_per_head;
  //int target_id = target_index(batch_id, seq_id, head_id, id, batch_size, seq_len, head_num, size_per_head);
  int bias_id = threadIdx.x;

  half2* src_ptr = (half2*)K;
  half2* dst_ptr = (half2*)k_buf_;
  const half2* bias_ptr = (const half2*)bias_K;
  half2 tmp;
  tmp = __hadd2(src_ptr[tid],  __ldg(&bias_ptr[bias_id]));
  if (do_lrelu)
  {
      tmp = my_leaky_relu<half2>(tmp, lrelu_alpha);
  }

  //for (int i = 0; i < batch_size; i++)
  {
      //int target_id = target_index(i, seq_id, head_id, id, batch_size, seq_len, head_num, size_per_head);
      int target_id =  id + seq_id * size_per_head + head_id * size_per_head * seq_len;
      dst_ptr[target_id] = tmp;
  }
}


template <typename T>
__global__
void add_QKV_bias_ln_KV_m1_V(T* V, const T* bias_V, T* v_buf_, const int batch_size,
        const int seq_len, const int head_num, const int size_per_head, const int word_per_block, bool do_lrelu, float lrelu_alpha)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  //int batch_id = blockIdx.x / seq_len;
  int seq_id = blockIdx.x % seq_len;
  int head_id =  threadIdx.x / size_per_head;
  int id = tid % size_per_head;
  //int target_id = target_index(batch_id, seq_id, head_id, id, batch_size, seq_len, head_num, size_per_head);
  int bias_id = threadIdx.x;

  float* src_ptr_v = (float*)V;
  float* dst_ptr_v = (float*)v_buf_;
  const float* bias_ptr = (const float*)bias_V;
  float tmp = src_ptr_v[tid] + bias_ptr[bias_id];
  if (do_lrelu)
  {
      tmp = my_leaky_relu<float>(tmp, lrelu_alpha);
  }
  //for (int i = 0; i < batch_size; i++)
  {
      //int target_id = target_index(i, seq_id, head_id, id, batch_size, seq_len, head_num, size_per_head);
      int target_id =  id + seq_id * size_per_head + head_id * size_per_head * seq_len;
      dst_ptr_v[target_id] = tmp;
  }
}

template <>
__global__
void add_QKV_bias_ln_KV_m1_V(half* V, const half* bias_V, half* v_buf_, const int batch_size,
        const int seq_len, const int head_num, const int size_per_head, const int word_per_block, bool do_lrelu, float lrelu_alpha)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  //int batch_id = blockIdx.x / seq_len;
  int seq_id = blockIdx.x % seq_len;
  int head_id =  threadIdx.x / size_per_head;
  int id = tid % size_per_head;
  //int target_id = target_index(batch_id, seq_id, head_id, id, batch_size, seq_len, head_num, size_per_head);
  int bias_id = threadIdx.x;

  half2* src_ptr_v = (half2*)V;
  half2* dst_ptr_v = (half2*)v_buf_;
  const half2* bias_ptr = (const half2*)bias_V;
  half2 tmp;
  tmp = __hadd2(src_ptr_v[tid],  __ldg(&bias_ptr[bias_id]));
  if (do_lrelu)
  {
      tmp = my_leaky_relu<half2>(tmp, lrelu_alpha);
  }
  //for (int i = 0; i < batch_size; i++)
  {
      //int target_id = target_index(i, seq_id, head_id, id, batch_size, seq_len, head_num, size_per_head);
      int target_id =  id + seq_id * size_per_head + head_id * size_per_head * seq_len;
      dst_ptr_v[target_id] = tmp;
  }
}

__device__
float bool2float(bool in)
{
    if(in == true)
        return 1.0f;
    else 
        return 0.0f;
}

template <typename T>
__global__
void softmax_kernel(const T* qk_buf_, T* inter_out, T* softmax, const bool* k_mask, const int batch_size,
        const int head_num, const int from_seq_len, const int to_seq_len, const float scalar)
{
    int batch_id = blockIdx.x / head_num;
    int qk_offset = blockIdx.x * from_seq_len * to_seq_len;
    int k_mask_offset = batch_id * to_seq_len;

    __shared__ float s_sum, s_max;

    for(int i = 0; i < from_seq_len; ++i)
    {
      float qk = threadIdx.x < to_seq_len ? (float)qk_buf_[threadIdx.x % to_seq_len + qk_offset] : 0.0f;
      bool mask_val = threadIdx.x < to_seq_len ? k_mask[threadIdx.x + k_mask_offset] : false;
      float padding = mask_val ? qk * scalar : qk * scalar - 10000.0f;
      float tmp = threadIdx.x < to_seq_len ? padding: 0.0f;

      float max_val = blockReduceMax<float>(tmp);

      if(threadIdx.x == 0)
        s_max = max_val;
      __syncthreads();

      qk = threadIdx.x < to_seq_len ? __expf(tmp - s_max) : 0.0f;

      float sum_val = blockReduceSum<float>(qk);

      if(threadIdx.x == 0)
      {
        s_sum = sum_val;
      }
      __syncthreads();

      if(threadIdx.x < to_seq_len)
      {
          T tmp_s;
          if (s_sum == 0.f)
              tmp_s = (T)(1.f / to_seq_len);
          else
              tmp_s = (T)(qk / s_sum);

          softmax[threadIdx.x + qk_offset] = tmp_s;
          inter_out[threadIdx.x + qk_offset] = tmp_s;
      }

      qk_offset += to_seq_len;
      //mask_offset += seq_len;
    }
}

template <>
__global__
void softmax_kernel(const half* qk_buf_, half* inter_out, half* softmax, const bool* k_mask, const int batch_size,
        const int head_num, const int from_seq_len, const int to_seq_len, const float scalar)
{
    int batch_id = blockIdx.x / head_num;
    int qk_offset = blockIdx.x * from_seq_len * to_seq_len;
    int k_mask_offset = batch_id * to_seq_len;

    __shared__ float s_sum, s_max;

    for(int i = 0; i < from_seq_len; ++i)
    {
      float qk = threadIdx.x < to_seq_len ? __half2float(qk_buf_[threadIdx.x % to_seq_len + qk_offset]) : 0.0f;
      bool mask_val = threadIdx.x < to_seq_len ? k_mask[threadIdx.x + k_mask_offset] : false;
      //float padding = mask_val ? qk * scalar : 1.0f - 2e32;
      float padding = mask_val ? qk * scalar : qk * scalar - 10000.0f;
      float tmp = threadIdx.x < to_seq_len ? padding: 0.0f;

      float max_val = blockReduceMax<float>(tmp);

      if(threadIdx.x == 0)
        s_max = max_val;
      __syncthreads();

      qk = threadIdx.x < to_seq_len ? __expf(tmp - s_max) : 0.0f;

      float sum_val = blockReduceSum<float>(qk);

      if(threadIdx.x == 0)
      {
        s_sum = sum_val;
      }
      __syncthreads();

      if(threadIdx.x < to_seq_len)
      {
          if (s_sum == 0.f)
              qk = 1.0f / to_seq_len;
          else 
              qk = qk / s_sum;

          softmax[threadIdx.x + qk_offset] = __float2half(qk);
          inter_out[threadIdx.x + qk_offset] = __float2half(qk);
      }

      qk_offset += to_seq_len;
      //mask_offset += seq_len;
    }
}

template <typename T>
__global__
void softmax_kernel_v2(const T* qk_buf_, T* inter_out, T* softmax, const bool* k_mask, const int batch_size,
        const int head_num, const int from_seq_len, const int to_seq_len, const float scalar)
{
    //int batch_id = (blockIdx.x / from_seq_len) % batch_size;
    int batch_id = blockIdx.x / (from_seq_len * head_num);
    //int seq_id = blockIdx.x % from_seq_len;
    int qk_offset = blockIdx.x * to_seq_len;
    int k_mask_offset = batch_id * to_seq_len;

    __shared__ float s_sum, s_max;

    float qk = threadIdx.x < to_seq_len ? (float)qk_buf_[threadIdx.x + qk_offset] : 0.0f;
    bool mask_val = threadIdx.x < to_seq_len ? k_mask[threadIdx.x + k_mask_offset] : false;
    float padding = mask_val ? qk * scalar : qk * scalar - 10000.0f;
    //float padding = mask_val ? qk * scalar : 1.0f - 2e32;
    float tmp = threadIdx.x < to_seq_len ? padding : 0.0f;
    float max_val = blockReduceMax<float>(tmp);
    if(threadIdx.x == 0)
      s_max = max_val;
    __syncthreads();

    float qk_tmp = threadIdx.x < to_seq_len ? __expf((float)(tmp - s_max)) : 0.0f;
    float sum_val = blockReduceSum<float>(qk_tmp);

    if(threadIdx.x == 0)
    {
      s_sum = sum_val;
    }
    __syncthreads();

    if(threadIdx.x < to_seq_len)
    {
        if (s_sum == 0.f)
            qk_tmp = 1.0f / to_seq_len;
        else 
            qk_tmp = qk_tmp / s_sum;

        softmax[threadIdx.x + qk_offset] = (T)qk_tmp;
        inter_out[threadIdx.x + qk_offset] = (T)qk_tmp;
    }
}

template <>
__global__
void softmax_kernel_v2(const half* qk_buf_, half* inter_out, half* softmax, const bool* k_mask, const int batch_size,
        const int head_num, const int from_seq_len, const int to_seq_len, const float scalar)
{
    //int head_id = (blockIdx.x / from_seq_len) % head_num;
    int batch_id = blockIdx.x / (from_seq_len * head_num);
    //int batch_id = (blockIdx.x / from_seq_len) % batch_size;
    //int seq_id = blockIdx.x % from_seq_len;
    int qk_offset = blockIdx.x * to_seq_len;
    int k_mask_offset = batch_id * to_seq_len;

    __shared__ float s_sum, s_max;

    float qk = threadIdx.x < to_seq_len ? __half2float(qk_buf_[threadIdx.x + qk_offset]) : 0.0f;
    bool mask_val = threadIdx.x < to_seq_len ? k_mask[threadIdx.x + k_mask_offset] : false;
    //float padding = mask_val ? qk * scalar : 1.0f - 2e32;
    float padding = mask_val ? qk * scalar : qk * scalar - 10000.0f;
    float tmp = threadIdx.x < to_seq_len ? padding : 0.0f;
    float max_val = blockReduceMax<float>(tmp);
    if(threadIdx.x == 0)
      s_max = max_val;
    __syncthreads();

    float qk_tmp = threadIdx.x < to_seq_len ? __expf((float)(tmp - s_max)) : 0.0f;
    float sum_val = blockReduceSum<float>(qk_tmp);

    if(threadIdx.x == 0)
    {
      s_sum = sum_val;
    }
    __syncthreads();

    if(threadIdx.x < to_seq_len)
    {
        if (s_sum == 0.f)
            qk_tmp = 1.0f/to_seq_len;
        else
            qk_tmp = qk_tmp / s_sum;

        softmax[threadIdx.x + qk_offset] = __float2half(qk_tmp);
        inter_out[threadIdx.x + qk_offset] = __float2half(qk_tmp);
    }
}

//grid = (seq_len, head_num, batch_size)
//block.x = max(32, (seq_len + 31)/32*32)
template <typename T>
__global__
void softmax_kernel_v3(const T* qk_buf_, T* output, T* softmax, const bool* k_mask, const int batch_size,
        const int head_num, const int from_seq_len, const int to_seq_len, const float scalar,
        const bool* mask, float dropout_rate)
{
    float tmp = 0.0f;
    int qk_offset;
    __shared__ float s_mean, s_max;
    if (threadIdx.x < to_seq_len){
        qk_offset = ((blockIdx.z*head_num + blockIdx.y)*from_seq_len + blockIdx.x) *to_seq_len + threadIdx.x;
        //qk_offset = ((blockIdx.z*batch_size + blockIdx.y)*from_seq_len + blockIdx.x) *to_seq_len + threadIdx.x;
        int k_mask_offset = blockIdx.z * to_seq_len + threadIdx.x;

        float qk = static_cast<float>(qk_buf_[qk_offset]);
        bool mask_val = k_mask[k_mask_offset];
        //tmp = mask_val ? qk * scalar : 1.0f - 2e32;
        tmp = mask_val ? qk * scalar : qk * scalar - 10000.0f;
    }

    float max_val = blockReduceMax<float>(tmp);
    if (threadIdx.x == 0){
        s_max = max_val;
    }
    __syncthreads();

    float qk_tmp = threadIdx.x < to_seq_len ? __expf(tmp - s_max) : 0.0f;
    float sum_val = blockReduceSum<float>(qk_tmp);
    if (threadIdx.x == 0){
        s_mean = sum_val;
    }
    __syncthreads();

    //curandState_t state;
    //if (dropout_rate > 0.0f)
    //curand_init(1024, qk_offset, 0, &state);

    if(threadIdx.x < to_seq_len)
    {
        if (s_mean < 1e-20 && s_mean > (0.0f-1e-20))
        {
            qk_tmp = 1.0f / to_seq_len;
        } else {
            qk_tmp = qk_tmp * __fdividef(1.0f, s_mean);
        }

        softmax[qk_offset] = (T)qk_tmp;

        if (dropout_rate > 0.0f) {
            bool m = mask[qk_offset];
            qk_tmp = m ? qk_tmp / (1.f - dropout_rate) : 0.0f;
        }

        output[qk_offset] = (T)qk_tmp;
    }
}

//grid = (seq_len, head_num, batch_size)
//block.x = max(32, (seq_len/2 + 31)/32*32)
//seq_len % 2 == 0
template <>
__global__
void softmax_kernel_v3(const half* qk_buf_, half* output, half* softmax, const bool* k_mask,
                      const int batch_size, const int head_num,
                      const int from_seq_len, const int to_seq_len, const float scalar,
                      const bool* mask, float rate)
{
    half2* qk_buf_half2Ptr = (half2*) qk_buf_;
    half2* out_buf_half2Ptr = (half2*) output;
    half2* softmax_half2Ptr = (half2*) softmax;

    int qk_offset;
    int threadIdx2 = threadIdx.x << 1;
    __shared__ float s_mean, s_max;
    half2 tmp = __float2half2_rn(0.0f);

    float max_val = 0.0f;
    half2 qk;
    if (threadIdx2 < to_seq_len){
        qk_offset = ((((blockIdx.z*head_num + blockIdx.y)*from_seq_len + blockIdx.x) *to_seq_len) >> 1) + threadIdx.x;
        //qk_offset = ((((blockIdx.z*batch_size + blockIdx.y)*from_seq_len + blockIdx.x) *to_seq_len) >> 1) + threadIdx.x;
        int k_mask_offset = (blockIdx.z * to_seq_len) + 2*threadIdx.x;

        qk = qk_buf_half2Ptr[qk_offset];
        float tmpf0 = 0.0f, tmpf1 = 0.0f;
        tmpf0 = k_mask[k_mask_offset] ? scalar * __half2float(qk.x) : scalar * __half2float(qk.x) - 10000.0f;
        tmpf1 = k_mask[k_mask_offset + 1] ? scalar * __half2float(qk.y) : scalar * __half2float(qk.y) - 10000.0f;
        //tmpf0 = k_mask[k_mask_offset] ? scalar * __half2float(qk.x) : 1.0f - 2e32;
        //tmpf1 = k_mask[k_mask_offset + 1] ? scalar * __half2float(qk.y) : 1.0f - 2e32;
        tmp.x = __float2half(tmpf0);
        tmp.y = __float2half(tmpf1);
        max_val = fmax(tmpf0, tmpf1);
    }

    max_val = blockDim.x <= 32 ? warpReduceMax(max_val) : blockReduceMax<float>(max_val);

    if (threadIdx.x == 0){
        s_max = max_val;
    }
    __syncthreads();

    if (threadIdx2 < to_seq_len){
        tmp = h2exp(__hsub2(tmp, __float2half2_rn(s_max)));
    }
    float sum_val = blockDim.x <= 32 ? warpReduceSum((float)(tmp.x + tmp.y)) : blockReduceSum<float>((float)(tmp.x + tmp.y));

    if (threadIdx.x == 0){
        s_mean = sum_val;
    }
    __syncthreads();

    if(threadIdx2 < to_seq_len){
        if (s_mean == 0.f)
        {
            qk = __float2half2_rn(1.0f / to_seq_len);
        }
        else {
            float tmp_s_mean = __fdividef(1.0f, s_mean);
            qk = __hmul2(tmp, __float2half2_rn(tmp_s_mean));
        }

        softmax_half2Ptr[qk_offset] = qk;
        out_buf_half2Ptr[qk_offset] = qk;
    }
}

//grid = (seq_len, head_num, batch_size)
//block.x = max(32, (seq_len + 31)/32*32)
//for seq_len not larger than 32
template <typename T>
__global__
void softmax_kernel_v3_LE32(const T* qk_buf_, T* inter_out, T* softmax, const bool* k_mask, const int batch_size,
        const int head_num, const int from_seq_len, const int to_seq_len, const float scalar)
{

    int qk_offset;
    __shared__ float s_mean, s_max;
    float tmp = 0.0f;
    if (threadIdx.x < to_seq_len){
        qk_offset = ((blockIdx.z*head_num + blockIdx.y)*from_seq_len + blockIdx.x) * to_seq_len + threadIdx.x;
        int k_mask_offset = blockIdx.z * to_seq_len + threadIdx.x;

        float qk = static_cast<float>(qk_buf_[qk_offset]);
        bool mask_val = k_mask[k_mask_offset];
        tmp = mask_val ? qk * scalar : qk * scalar - 10000.0f;
        //tmp = mask_val ? qk * scalar : 1.0f - 2e32;
    }
    float max_val = warpReduceMax<float>(tmp);

    if (threadIdx.x == 0){
      s_max = max_val;
    }
    __syncthreads();

    tmp = threadIdx.x < to_seq_len ? __expf(tmp - s_max) : 0.0f;
    float sum_val = warpReduceSum<float>(tmp);

    if (threadIdx.x == 0){
      s_mean = sum_val;
    }
    __syncthreads();

    if(threadIdx.x < to_seq_len)
    {
        if (s_mean == 0.f)
        {
            tmp = 1.0f / to_seq_len;
        } else {
            tmp = tmp * __fdividef(1.0f, s_mean);
        }

        softmax[qk_offset] = (T)tmp;
        inter_out[qk_offset] = (T)tmp;
    }
}

template <>
__global__
void softmax_kernel_v3_LE32(const half* qk_buf_, half* inter_out, half* softmax, const bool* k_mask, const int batch_size,
        const int head_num, const int from_seq_len, const int to_seq_len, const float scalar)
{

    int qk_offset;
    __shared__ float s_mean, s_max;
    float tmp = 0.0f;
    if (threadIdx.x < to_seq_len){
        qk_offset = ((blockIdx.z*head_num + blockIdx.y)*from_seq_len + blockIdx.x) * to_seq_len + threadIdx.x;
        int k_mask_offset = blockIdx.z * to_seq_len + threadIdx.x;

        float qk = __half2float(qk_buf_[qk_offset]);
        bool mask_val = k_mask[k_mask_offset];
        tmp = mask_val ? __half2float(qk) * scalar : __half2float(qk) * scalar - 10000.0f;
        //tmp = mask_val ? __half2float(qk) * scalar : 1.0f - 2e32;
    }
    float max_val = warpReduceMax<float>(tmp);

    if (threadIdx.x == 0){
      s_max = max_val;
    }
    __syncthreads();

    tmp = threadIdx.x < to_seq_len ? __expf(tmp - s_max) : 0.0f;
    float sum_val = warpReduceSum<float>(tmp);

    if (threadIdx.x == 0){
      s_mean = sum_val;
    }
    __syncthreads();

    if(threadIdx.x < to_seq_len)
    {
        if (s_mean == 0.f)
        {
            tmp = 1.0f / to_seq_len;
        } else {
            tmp = tmp * __fdividef(1.0f, s_mean);
        }

        softmax[qk_offset] = tmp;
        inter_out[qk_offset] = tmp;
    }
}

template<typename T>
__global__
void transpose(T* src, T* dst, const int batch_size, const int seq_len, const int head_num, const int size_per_head)
{
  int batch_id = (blockIdx.x / seq_len) % batch_size;
  int seq_id = blockIdx.x % seq_len;
  int head_id = blockIdx.x / (batch_size * seq_len);
  dst[batch_id * (head_num * seq_len * size_per_head) + seq_id * head_num * size_per_head
    + head_id * size_per_head + threadIdx.x] = src[blockIdx.x * size_per_head + threadIdx.x];
}

template<>
  __global__
void transpose(half* src, half* dst,
    const int batch_size, const int seq_len, const int head_num, const int size_per_head)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int batch_id = tid / (seq_len * size_per_head) % batch_size;
  //int batch_id = tid / (head_num * seq_len * size_per_head);
  //int head_id = (tid % (head_num * seq_len * size_per_head)) / (seq_len * size_per_head);
  int head_id = tid / (batch_size * seq_len * size_per_head);
  int seq_id = (tid % (seq_len * size_per_head)) / size_per_head;
  int id = tid % size_per_head;

  int target_id = target_index(batch_id, head_id, seq_id, id, batch_size, head_num, seq_len, size_per_head);
  half2* src_ptr = (half2*)src;
  half2* dst_ptr = (half2*)dst;

  dst_ptr[target_id] = src_ptr[tid];
}


template<typename T>
__global__
void transpose_rebuild_padding(T* src, T* dst, const int batch_size, const int seq_len, const int head_num, const int size_per_head,
  const int* mask_offset)
{
  // TODO: optimize this kernel? 
  // do remove_sequence_length_padding
  const int tid = threadIdx.x; // batch * seq_len
  const int bid = blockIdx.x; // head_num * size_per_head

  const int src_batch_id = (bid + mask_offset[bid]) / seq_len;
  const int src_seq_id = (bid + mask_offset[bid]) % seq_len;

  const int dst_seq_id = bid;

  const int head_id = tid / size_per_head;
  const int hidden_id = tid % size_per_head;
  dst[dst_seq_id * head_num * size_per_head + tid] = src[ src_batch_id * head_num * seq_len * size_per_head +
    head_id * seq_len * size_per_head + src_seq_id * size_per_head + hidden_id];
}

template<typename T>
__global__ void rebuild_sequence_length_padding(const T* src, T* tgt,
                                            const int* mask_offset,
                                            const int n)
{
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int tgt_seq_id = bid + mask_offset[bid];
  const int src_seq_id = bid;

  for(int i = tid; i < n; i += blockDim.x)
  {
    tgt[tgt_seq_id * n + i] = src[src_seq_id * n + i];
  }
}

void print_to_file_float(float* result, const int size, char* file)
{
  FILE* fd = fopen(file, "w");
  float* tmp = (float*)malloc(sizeof(float) * size);
  cudaMemcpy(tmp, result, sizeof(float) * size, cudaMemcpyDeviceToHost);
  for(int i = 0; i < size; ++i)
    fprintf(fd, "%f\n", tmp[i]);
  free(tmp);
  fclose(fd);
}

__global__
void half2float(half* input, float* output, int size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size)
    {
        float tmp = __half2float(input[tid]);
        output[tid] = tmp;
    }
}

void print_to_file_half(half* result, const int size, char* file)
{
  FILE* fd = fopen(file, "w");
  float* tmp = (float*)malloc(sizeof(float) * size);
  float* d_ptr;
  cudaMalloc((void **)&d_ptr, sizeof(float)*size);

  int block = 512;
  int grid = (size + 511) / 512;
  half2float<<<grid, block>>>(result, d_ptr, size);

  cudaMemcpy(tmp, d_ptr, sizeof(float) * size, cudaMemcpyDeviceToHost);
  for(int i = 0; i < size; ++i)
    fprintf(fd, "%f\n", tmp[i]);

  cudaFree(d_ptr);
  free(tmp);
  fclose(fd);
}

template<OperationType OpType_>
void OpenMultiHeadAttention<OpType_>::multiHeadAttr_nofuse_kernelLauncher(
      cudaStream_t stream,
      const int batch_size,
      const int from_seq_len,
      const int to_seq_len,
      const int head_num,
      const float scalar)
{
    dim3 grid;
    dim3 block;

//#    size_t size = batch_size*from_seq_len*to_seq_len*head_num;
//#    cudaMemcpy(param_.inter_output, param_.attention_scores, sizeof(float)*size, cudaMemcpyDeviceToHost);
//#
//#    printf("############# FILE: %s, LINE: %d\n", __FILE__, __LINE__);
//#    if (OpType_ == OperationType::FP32) {
//#        //save_gpu_float((const float *)param_.attention_scores, batch_size*from_seq_len*to_seq_len*head_num);
//#        //save_gpu_bool(param_.k_mask, batch_size*from_seq_len*to_seq_len*head_num);
//#
//#        size_t size = batch_size*to_seq_len;
//#        bool * h_ptr = (bool *)malloc(sizeof(bool) * size);
//#        FILE *fp = NULL;
//#
//#        cudaMemcpy(h_ptr, param_.k_mask, sizeof(bool) *size, cudaMemcpyDeviceToHost);
//#
//#        fp = fopen("bool_data.txt", "w+");
//#        for (size_t i = 0; i < size; i++) {
//#            bool value = h_ptr[i];
//#            if (value)
//#                fprintf(fp, "%d\n", 1);
//#            else
//#                fprintf(fp, "%d\n", 0);
//#        }
//#
//#        fclose(fp);
//#        free(h_ptr);
//#    }

    //deal with odd to_seq_len
    if (to_seq_len % 2 != 0){
      if(to_seq_len <= 32)
        block.x = 32;
      else if(to_seq_len > 32 && to_seq_len <= 64)
        block.x = 64;
      else if(to_seq_len > 64 && to_seq_len <= 128)
        block.x = 128;
      else if(to_seq_len > 128 && to_seq_len <= 256)
        block.x = 256;
      else if(to_seq_len > 256 && to_seq_len <= 512)
        block.x = 512;
      else
        block.x = 1024;

      //data shape [batch, head_num, from_seq_len, to_seq_len]

      if(batch_size * head_num <= 120)
      {
        grid.x = batch_size * head_num * from_seq_len;
        if (OpType_ == OperationType::FP16) {
            softmax_kernel_v2<<<grid, block, 0, stream>>>((half *)param_.attention_scores, (half*)param_.output, (half*)param_.softmax, param_.k_mask, 
                    batch_size, head_num, from_seq_len, to_seq_len, scalar);
        } else {
            softmax_kernel_v2<DataType_><<<grid, block, 0, stream>>>(param_.attention_scores, param_.output, param_.softmax, param_.k_mask, 
                    batch_size, head_num, from_seq_len, to_seq_len, scalar);
        }
        add_logging("Logging: computing softmax and masking (kernel 0)... ", __FILE__, __LINE__);
      }
      else
      {
        grid.x = batch_size * head_num;
        if (OpType_ == OperationType::FP16) {
            softmax_kernel<<<grid, block, 0, stream>>>((half*)param_.attention_scores, (half*)param_.output, (half*)param_.softmax, param_.k_mask, 
                    batch_size, head_num, from_seq_len, to_seq_len, scalar);
        } else {
            softmax_kernel<DataType_><<<grid, block, 0, stream>>>(param_.attention_scores, param_.output, param_.softmax, param_.k_mask, 
                    batch_size, head_num, from_seq_len, to_seq_len, scalar);
        }
        add_logging("Logging: computing softmax and masking (kernel 1)... ", __FILE__, __LINE__);
      }
    }
    //deal with even seq_len
    else{
      grid.x = from_seq_len;
      grid.y = head_num;
      grid.z = batch_size;
      if (to_seq_len <= 32){
        block.x = 32;
        if (OpType_ == OperationType::FP16){
            softmax_kernel_v3_LE32<<<grid, block, 0, stream>>>((half *)param_.attention_scores, (half*)param_.output, (half*)param_.softmax, param_.k_mask, 
                    batch_size, head_num, from_seq_len, to_seq_len, scalar);
        } else {
            softmax_kernel_v3_LE32<DataType_><<<grid, block, 0, stream>>>(param_.attention_scores, param_.output, param_.softmax, param_.k_mask, 
                    batch_size, head_num, from_seq_len, to_seq_len, scalar);
        }
        add_logging("Logging: computing softmax and masking (kernel 2)... ", __FILE__, __LINE__);
      }
      else{
          //printf("dropout_rate = %f\n", param_.dropout_rate);
          //printf("dropout_rate = %f\n", param_.dropout_rate);
          //printf("dropout_rate = %f\n", param_.dropout_rate);
          //printf("dropout_rate = %f\n", param_.dropout_rate);
          //printf("dropout_rate = %f\n", param_.dropout_rate);
          //printf("dropout_rate = %f\n", param_.dropout_rate);
        //std::pair<uint64_t, uint64_t> seed = Context::Instance().IncrementOffset(1);
        if (OpType_ == OperationType::FP16){
         //TODO: curand is not implemented for fp16
          block.x = (to_seq_len/2 + 31)/32*32;
          softmax_kernel_v3<<<grid, block, 0, stream>>>((half*)param_.attention_scores, (half*)param_.output, (half*)param_.softmax, param_.k_mask, 
                  batch_size, head_num, from_seq_len, to_seq_len, scalar, param_.mask, param_.dropout_rate);
        add_logging("Logging: computing softmax and masking (kernel 3)... ", __FILE__, __LINE__);
        }
        else{
          block.x = (to_seq_len + 31)/32*32;
          softmax_kernel_v3<DataType_><<<grid, block, 0, stream>>>(param_.attention_scores,param_.output, param_.softmax, param_.k_mask, 
                  batch_size, head_num, from_seq_len, to_seq_len, scalar, param_.mask, param_.dropout_rate);
        add_logging("Logging: computing softmax and masking (kernel 4)... ", __FILE__, __LINE__);
        }
      }
      grid.x = grid.y = grid.z = 1;
    }
}

template void OpenMultiHeadAttention<OperationType::FP32>::multiHeadAttr_nofuse_kernelLauncher(
      cudaStream_t stream,
      const int batch_size,
      const int from_seq_len,
      const int to_seq_len,
      const int head_num,
      const float scalar);

template void OpenMultiHeadAttention<OperationType::FP16>::multiHeadAttr_nofuse_kernelLauncher(
      cudaStream_t stream,
      const int batch_size,
      const int from_seq_len,
      const int to_seq_len,
      const int head_num,
      const float scalar);
}//namespace cuda
}//namespace multiheadattention
