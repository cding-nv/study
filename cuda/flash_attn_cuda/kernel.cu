// flash_attn.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void flash_attn_forward_kernel(
    const c10::Half* __restrict__ q,
    const c10::Half* __restrict__ k,
    const c10::Half* __restrict__ v,
    c10::Half* __restrict__ out,
    int B, int T, int H, int D
) {
    int b = blockIdx.x;
    int t = threadIdx.x;

    if (t >= T) return;

    for (int h = 0; h < H; ++h) {
        for (int d = 0; d < D; ++d) {
            float acc = 0.0f;
            float sum_exp = 0.0f;

            for (int tp = 0; tp < T; ++tp) {
                float dot = 0.0f;
                for (int i = 0; i < D; ++i) {
                    int q_idx = (((b * T + t) * H + h) * D) + i;
                    int k_idx = (((b * T + tp) * H + h) * D) + i;
                    dot += __half2float(q[q_idx]) * __half2float(k[k_idx]);
                }
                dot /= sqrtf((float)D);
                float weight = expf(dot);
                sum_exp += weight;

                int v_idx = (((b * T + tp) * H + h) * D) + d;
                acc += weight * __half2float(v[v_idx]);
            }

            int o_idx = (((b * T + t) * H + h) * D) + d;
            out[o_idx] = __float2half(acc / sum_exp);  // softmax normalization
        }
    }
}

void flash_attn_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor out) {
    int B = q.size(0), T = q.size(1), H = q.size(2), D = q.size(3);
    const int threads = 128;
    flash_attn_forward_kernel<<<B, threads>>>(
        q.data_ptr<at::Half>(),
        k.data_ptr<at::Half>(),
        v.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(),
        B, T, H, D
    );
}

#define BLOCK_SIZE 16

// __global__ void flash_attn_blockwise_kernel(
//     const half* __restrict__ q,
//     const half* __restrict__ k,
//     const half* __restrict__ v,
//     half* __restrict__ out,
//     int B, int T, int H, int D
// ) {
//     extern __shared__ float shared[];

//     int b = blockIdx.x;      // batch
//     int h = blockIdx.y;      // head
//     int block_t = blockIdx.z * BLOCK_SIZE;  // start token index of this block
//     int t = threadIdx.x;     // token index within block

//     if (block_t + t >= T) return;

//     float* q_vec = new float[D];
//     float* acc = new float[D];
//     for (int i = 0; i < D; ++i) acc[i] = 0.0f;
//     float sum_exp = 0.0f;

//     // Load query vector q[b, block_t + t, h, :]
//     for (int d = 0; d < D; ++d) {
//         int q_idx = (((b * T + (block_t + t)) * H + h) * D) + d;
//         q_vec[d] = __half2float(q[q_idx]);
//     }

//     for (int tp_block = 0; tp_block < T; tp_block += BLOCK_SIZE) {
//         __shared__ float k_block[BLOCK_SIZE][256];
//         __shared__ float v_block[BLOCK_SIZE][256];

//         int tp = tp_block + threadIdx.x;
//         if (tp < T) {
//             for (int d = threadIdx.y; d < D; d += blockDim.y) {
//                 if (d < 256) {
//                     int k_idx = (((b * T + tp) * H + h) * D) + d;
//                     int v_idx = k_idx;
//                     k_block[threadIdx.x][d] = __half2float(k[k_idx]);
//                     v_block[threadIdx.x][d] = __half2float(v[v_idx]);
//                 }
//             }
//         }
//         __syncthreads();

//         for (int i = 0; i < BLOCK_SIZE && (tp_block + i) < T; ++i) {
//             float dot = 0.0f;
//             for (int d = 0; d < D; ++d) {
//                 dot += q_vec[d] * k_block[i][d];
//             }
//             float scaled = dot / sqrtf((float)D);
//             float weight = expf(scaled);
//             sum_exp += weight;
//             for (int d = 0; d < D; ++d) {
//                 acc[d] += weight * v_block[i][d];
//             }
//         }
//         __syncthreads();
//     }

//     for (int d = 0; d < D; ++d) {
//         int o_idx = (((b * T + (block_t + t)) * H + h) * D) + d;
//         out[o_idx] = __float2half(acc[d] / sum_exp);
//     }

//     delete[] q_vec;
//     delete[] acc;
// }

// void flash_attn_blockwise_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor out) {
//     int B = q.size(0), T = q.size(1), H = q.size(2), D = q.size(3);
//     dim3 blocks(B, H, (T + BLOCK_SIZE - 1) / BLOCK_SIZE);
//     dim3 threads(BLOCK_SIZE, 1);
//     size_t shared_mem = 2 * BLOCK_SIZE * D * sizeof(float);

//     flash_attn_blockwise_kernel<<<blocks, threads, shared_mem>>>(
//         reinterpret_cast<half*>(q.data_ptr<at::Half>()),
//         reinterpret_cast<half*>(k.data_ptr<at::Half>()),
//         reinterpret_cast<half*>(v.data_ptr<at::Half>()),
//         reinterpret_cast<half*>(out.data_ptr<at::Half>()),
//         B, T, H, D
//     );
// }

//###### this is ok ##### 

__global__ void flash_attn_blockwise_kernel(
    const half* __restrict__ q,
    const half* __restrict__ k,
    const half* __restrict__ v,
    half* __restrict__ out,
    int B, int T, int H, int D
) {
    int b = blockIdx.x;      // batch
    int h = blockIdx.y;      // head
    int t = threadIdx.x;     // time step index within BLOCK_SIZE
    int d = threadIdx.y;     // dimension index

    int block_start = blockIdx.z * BLOCK_SIZE;
    if (block_start + t >= T || d >= D) return;

    // Load full q vector (D-dim)  每个 thread 加载一个完整的 Q 向量 
    float q_vec[128];  // assuming D <= 128
    for (int i = 0; i < D; ++i) {
        q_vec[i] = __half2float(q[(((b * T + (block_start + t)) * H + h) * D) + i]);
    }

    // acc, l_i, m_i 最终累积的 值和，权重和，最大值，用于最终归一化
    float acc[128] = {0.0f};
    float l_i = 0.0f;
    float m_i = -INFINITY;

    for (int tp_block = 0; tp_block < T; tp_block += BLOCK_SIZE) {
        __shared__ float k_block[BLOCK_SIZE][128];
        __shared__ float v_block[BLOCK_SIZE][128];

        // 每个 thread 加载 block size 个完整 Key 和 Value
        if (tp_block + t < T && d < D) {
            k_block[t][d] = __half2float(k[(((b * T + (tp_block + t)) * H + h) * D) + d]);
            v_block[t][d] = __half2float(v[(((b * T + (tp_block + t)) * H + h) * D) + d]);
        }
        __syncthreads();

        // m_ij 当前 block 的 softmax 最大值
        // l_ij 当前 block softmax 之和
        // acc_block 当前 block 和 value 点乘之和
        // 每个 thread 计算 一个 Q 向量 和 block size 个 Key 向量点乘，并记录最大值 m_ij
        float m_ij = -INFINITY;
        for (int i = 0; i < BLOCK_SIZE && (tp_block + i) < T; ++i) {
            float dot = 0.0f;
            for (int j = 0; j < D; ++j) {
                dot += q_vec[j] * k_block[i][j];
            }
            float scaled = dot / sqrtf((float)D);
            m_ij = fmaxf(m_ij, scaled);
        }
        __syncthreads();

        float l_ij = 0.0f;
        float acc_block[128] = {0.0f};

        // 有 qk 重复计算
        for (int i = 0; i < BLOCK_SIZE && (tp_block + i) < T; ++i) {
            float dot = 0.0f;
            for (int j = 0; j < D; ++j) {
                dot += q_vec[j] * k_block[i][j];
            }
            float scaled = dot / sqrtf((float)D);
            float exp_val = expf(scaled - m_ij);
            l_ij += exp_val;
            acc_block[d] += exp_val * v_block[i][d];
        }
        __syncthreads();

        float m_new = fmaxf(m_i, m_ij);
        float alpha = expf(m_i - m_new);
        float beta = expf(m_ij - m_new);

        acc[d] = alpha * acc[d] + beta * acc_block[d];
        l_i = alpha * l_i + beta * l_ij;
        m_i = m_new;
    }

    // 每个 thread 回写 最终 out[b, t, h, d] 中的一个
    int o_idx = (((b * T + (block_start + t)) * H + h) * D) + d;
    out[o_idx] = __float2half(acc[d] / l_i);
}

void flash_attn_blockwise_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor out) {
    int B = q.size(0), T = q.size(1), H = q.size(2), D = q.size(3);
    dim3 blocks(B, H, (T + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threads(BLOCK_SIZE, D);

    flash_attn_blockwise_kernel<<<blocks, threads>>>(
        reinterpret_cast<half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        B, T, H, D
    );
}

// __global__ void flash_attn_blockwise_kernel(
//     const half* __restrict__ q,
//     const half* __restrict__ k,
//     const half* __restrict__ v,
//     half* __restrict__ out,
//     int B, int T, int H, int D
// ) {
//     int b = blockIdx.x;      // batch
//     int h = blockIdx.y;      // head
//     int t = threadIdx.x;     // time step index within BLOCK_SIZE
//     int d = threadIdx.y;     // dimension index

//     int block_start = blockIdx.z * BLOCK_SIZE;
//     if (block_start + t >= T || d >= D) return;

//     // Load full q vector (D-dim)
//     float q_vec[BLOCK_SIZE];
//     for (int i = 0; i < D; ++i) {
//         q_vec[i] = __half2float(q[(((b * T + (block_start + t)) * H + h) * D) + i]);
//     }

//     float acc = 0.0f;
//     float l_i = 0.0f;
//     float m_i = -INFINITY;

//     for (int tp_block = 0; tp_block < T; tp_block += BLOCK_SIZE) {
//         __shared__ float k_block[BLOCK_SIZE][BLOCK_SIZE];
//         __shared__ float v_block[BLOCK_SIZE][BLOCK_SIZE];

//         if (tp_block + t < T && d < D) {
//             k_block[t][d] = __half2float(k[(((b * T + (tp_block + t)) * H + h) * D) + d]);
//             v_block[t][d] = __half2float(v[(((b * T + (tp_block + t)) * H + h) * D) + d]);
//         }
//         __syncthreads();

//         float m_ij = -INFINITY;
//         float l_ij = 0.0f;
//         float acc_block = 0.0f;
//         float scaled_logits[BLOCK_SIZE];
//         int valid_len = 0;

//         for (int i = 0; i < BLOCK_SIZE && (tp_block + i) < T; ++i) {
//             float dot = 0.0f;
//             for (int j = 0; j < D; ++j) {
//                 dot += q_vec[j] * k_block[i][j];
//             }
//             float scaled = dot / sqrtf((float)D);
//             scaled_logits[i] = scaled;
//             m_ij = fmaxf(m_ij, scaled);
//             valid_len++;
//         }
//         __syncthreads();

//         for (int i = 0; i < valid_len; ++i) {
//             float exp_val = expf(scaled_logits[i] - m_ij);
//             l_ij += exp_val;
//             acc_block += exp_val * v_block[i][d];
//         }
//         __syncthreads();

//         float m_new = fmaxf(m_i, m_ij);
//         float alpha = expf(m_i - m_new);
//         float beta = expf(m_ij - m_new);

//         acc = alpha * acc + beta * acc_block;
//         l_i = alpha * l_i + beta * l_ij;
//         m_i = m_new;
//     }

//     int o_idx = (((b * T + (block_start + t)) * H + h) * D) + d;
//     out[o_idx] = __float2half(acc / l_i);
// }

// void flash_attn_blockwise_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor out) {
//     int B = q.size(0), T = q.size(1), H = q.size(2), D = q.size(3);
//     dim3 blocks(B, H, (T + BLOCK_SIZE - 1) / BLOCK_SIZE);
//     dim3 threads(BLOCK_SIZE, D);

//     flash_attn_blockwise_kernel<<<blocks, threads>>>(
//         reinterpret_cast<half*>(q.data_ptr<at::Half>()),
//         reinterpret_cast<half*>(k.data_ptr<at::Half>()),
//         reinterpret_cast<half*>(v.data_ptr<at::Half>()),
//         reinterpret_cast<half*>(out.data_ptr<at::Half>()),
//         B, T, H, D
//     );
// }