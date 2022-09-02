#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>

#include <cooperative_groups.h>
#include <curand_kernel.h>

#define MAX_THREADS 1024
#define THREADS 256

#define MAX_THREAD_STRIDE 32
#define TILE_DIM 32

// Maximum sequence-length support based on the number of threads (2048) allowed in each block and
// this MAX is 8K For higher sequence length we need to use higher Max, like for 64K : 32
#define MAX_THREAD_ITERATIONS 8  // Maximum 8K
#define MAX_WARP_NUM 32

#define MAX_REGISTERS 256


#define minus_infinity -1 * std::numeric_limits<float>::infinity()

#define FINAL_MASK 0xffffffff

#define CUDA_1D_KERNEL_LOOP(i, n) \
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

template<typename T>
void launch_transpose1203(const T* src, T* dst,
                             const int batch_size,
                             const int seq_len,
                             const int head_num,
                             const int size_per_head,
                             cudaStream_t& stream);

template<typename T>
void launch_transpose2013(const T* src, T* dst,
                             const int batch_size,
                             const int seq_len,
                             const int head_num,
                             const int size_per_head,
                             cudaStream_t& stream);

template<typename T>
void launch_transpose1203_lreluback(const T* src, T* dst,
                             const T* value_layer,
                             float alpha,
                             const int batch_size,
                             const int seq_len,
                             const int head_num,
                             const int size_per_head,
                             cudaStream_t& stream);

template <typename T>
void launch_backprop_masking_softmax(T *d_attention, const T* tf_softmax,
        const bool* mask, T* d_score,
        int head_num, int bsz, int from_seq_len,
        int to_seq_len, int size_per_head, float rate,
        cudaStream_t& stream);

// Custom fused bias add with layer normalization
template <typename T>
void launch_bias_residual_layer_norm(T* vals,
                                     const T* residual,
                                     const T* gamma,
                                     const T* beta,
                                     float epsilon,
                                     int batch_size,
                                     int hidden_dim,
                                     cudaStream_t stream,
                                     bool preLayerNorm,
                                     bool training,
                                     T* vars,
                                     T* means);

template <typename T>
void launch_bias_residual_layer_norm(T* vals,
                                     const T* residual,
                                     const T* gamma,
                                     const T* beta,
                                     float epsilon,
                                     int batch_size,
                                     int hidden_dim,
                                     cudaStream_t stream,
                                     bool preLayerNorm,
                                     bool training,
                                     T* vars);

template <typename T>
void launch_layerNorm_backward_fused_add(const T* out_grad1,
                                         const T* out_grad2,
                                         const T* X_data,
                                         const T* vars,
                                         const T* means,
                                         const T* gamma,
                                         T* gamma_grad,
                                         T* betta_grad,
                                         T* inp_grad,
                                         int batch_size,
                                         int hidden_dim,
                                         cudaStream_t stream);
template <typename T>
void launch_layerNorm_backward_fused_add(const T* out_grad1,
                                         const T* out_grad2,
                                         const T* vals_hat,
                                         const T* vars,
                                         const T* gamma,
                                         T* gamma_grad,
                                         T* betta_grad,
                                         T* inp_grad,
                                         int batch_size,
                                         int hidden_dim,
                                         cudaStream_t stream,
                                         bool invertible = false,
                                         const T* betta = nullptr);

template <typename T>
void launch_layerNorm_backward(const T* out_grad,
                               const T* X_data,
                               const T* vars,
                               const T* means,
                               const T* gamma,
                               T* gamma_inter,
                               T* betta_inter,
                               T* gamma_grad,
                               T* betta_grad,
                               T* inp_grad,
                               int batch_size,
                               const int hidden_dim,
                               const int B,
                               const int H,
                               const int S,
                               const float alpha,
                               cudaStream_t stream);

template <typename T>
void launch_layerNorm_backward(const T* out_grad,
                               const T* vals_hat,
                               const T* vars,
                               const T* gamma,
                               T* gamma_grad,
                               T* betta_grad,
                               T* inp_grad,
                               int batch_size,
                               int hidden_dim,
                               cudaStream_t stream,
                               bool invertible = false,
                               const T* betta = nullptr);


template <typename T>
void launch_sum_reduce_w(const T* inp_q,
                         T* out_q,
                         int batch,
                         int C,
                         int C_q,
                         cudaStream_t stream);


template <typename T>
void launch_column_sum_reduce(const T* inp,
                                       T* out,
                                       int rows, // N * T
                                       int cols, // C
                                       cudaStream_t stream);

