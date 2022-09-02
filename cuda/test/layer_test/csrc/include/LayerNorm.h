#pragma once
#include "errMsg.h"
#include "utils.h"
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <typeinfo>
namespace cg = cooperative_groups;


template <int iterCnt, int stride>
__global__ void
layernorm(int nCol,
          float epsilon,
          const float* __restrict__ inp,
          const float* __restrict__ gamma,
          const float* __restrict__ beta,
          float* __restrict__ out,
          bool training,
          float* __restrict__ rstd) {
    constexpr int nWarp = stride >> 5;      // power of 2
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);

    inp += blockIdx.x * nCol;
    out += blockIdx.x * nCol;

    float buff[iterCnt];
    float sumE = 0.f;
    float sumV = 0.f;
    for (int it = 0; it < iterCnt; ++it) {
        int idx = threadIdx.x + it * stride;
        if (idx >= nCol) break;
        sumE += buff[it] = inp[idx];
        sumV += square(buff[it]);
    }
    for (int i = 1; i < 32; i <<= 1) {
        sumE += tile.shfl_xor(sumE, i);
        sumV += tile.shfl_xor(sumV, i);
    }

    __shared__ float smE[nWarp];
    __shared__ float smV[nWarp];
    if (tile.thread_rank() == 0) {
        smE[threadIdx.x >> 5] = sumE;
        smV[threadIdx.x >> 5] = sumV;
    }
    block.sync();

    if (tile.thread_rank() < nWarp) {
        sumE = smE[tile.thread_rank()];
        sumV = smV[tile.thread_rank()];
    }
    for (int i = 1; i < nWarp; i <<= 1) {
        sumE += tile.shfl_xor(sumE, i);
        sumV += tile.shfl_xor(sumV, i);
    }
    sumE = tile.shfl(sumE, 0);
    sumV = tile.shfl(sumV, 0);

    float meanF = sumE / nCol;
    float rstdF = rsqrtf(sumV / nCol - square(meanF) + epsilon);

    for (int it = 0; it < iterCnt; ++it) {
        int idx = threadIdx.x + it * stride;
        if (idx >= nCol) break;
        out[idx] = (buff[it] - meanF) * rstdF * gamma[idx] + beta[idx];
    }
    if (training && threadIdx.x == 0) {
        rstd[blockIdx.x] = rstdF;
    }
}


template <int iterCnt, int stride>
__global__ void
layernorm(int nCol,
          float epsilon,
          const __half* __restrict__ inp,
          const __half* __restrict__ gamma,
          const __half* __restrict__ beta,
          __half* __restrict__ out,
          bool training,
          __half* __restrict__ rstd) {
    constexpr int nWarp = stride >> 5;      // power of 2
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);

    const __half2* inp2 = reinterpret_cast<const __half2*>(inp) + blockIdx.x * nCol;
    const __half2* gamma2 = reinterpret_cast<const __half2*>(gamma);
    const __half2* beta2 = reinterpret_cast<const __half2*>(beta);
    __half2* out2 = reinterpret_cast<__half2*>(out) + blockIdx.x * nCol;

    float2 buff2[iterCnt];
    float sumE = 0.f;
    float sumV = 0.f;
    for (int it = 0; it < iterCnt; ++it) {
        int idx = threadIdx.x + it * stride;
        if (idx >= nCol) break;
        buff2[it] = __half22float2(inp2[idx]);
        sumE += buff2[it].x + buff2[it].y;
        sumV += square(buff2[it].x) + square(buff2[it].y);
    }
    for (int i = 1; i < 32; i <<= 1) {
        sumE += tile.shfl_xor(sumE, i);
        sumV += tile.shfl_xor(sumV, i);
    }

    __shared__ float smE[nWarp];
    __shared__ float smV[nWarp];
    if (tile.thread_rank() == 0) {
        smE[threadIdx.x >> 5] = sumE;
        smV[threadIdx.x >> 5] = sumV;
    }
    block.sync();

    if (tile.thread_rank() < nWarp) {
        sumE = smE[tile.thread_rank()];
        sumV = smV[tile.thread_rank()];
    }
    for (int i = 1; i < nWarp; i <<= 1) {
        sumE += tile.shfl_xor(sumE, i);
        sumV += tile.shfl_xor(sumV, i);
    }
    sumE = tile.shfl(sumE, 0);
    sumV = tile.shfl(sumV, 0);

    sumE /= nCol << 1;
    sumV /= nCol << 1;
    __half2 mean2 = __float2half2_rn(sumE);
    __half2 rstd2 = __float2half2_rn(rsqrtf(sumV - square(sumE) + epsilon));

    for (int it = 0; it < iterCnt; ++it) {
        int idx = threadIdx.x + it * stride;
        if (idx >= nCol) break;
        out2[idx] = (__float22half2_rn(buff2[it]) - mean2) * rstd2 * gamma2[idx] + beta2[idx];
    }
    if (training && threadIdx.x == 0) {
        rstd[blockIdx.x] = rstd2.x;
    }
}


// out-of-place layernorm forward
template <typename T>
inline void
launchLayernorm(int nRow,
                int nCol,
                float epsilon,
                const T* inp,
                const T* gamma,
                const T* beta,
                T* out,
                bool training,
                T* rstd = nullptr,
                cudaStream_t stream = 0) {
    if (typeid(T) == typeid(__half)) {
        nCol >>= 1;
    }
    if (nCol <= 128) {
        layernorm<2, 64><<<nRow, 64, 0, stream>>>(nCol, epsilon, inp, gamma, beta, out, training, rstd);
    }
    else if (nCol <= 256) {
        layernorm<4, 64><<<nRow, 64, 0, stream>>>(nCol, epsilon, inp, gamma, beta, out, training, rstd);
    }
    else if (nCol <= 512) {
        layernorm<8, 64><<<nRow, 64, 0, stream>>>(nCol, epsilon, inp, gamma, beta, out, training, rstd);
    }
    else if (nCol <= 1024) {
        layernorm<8, 128><<<nRow, 128, 0, stream>>>(nCol, epsilon, inp, gamma, beta, out, training, rstd);
    }
    else if (nCol <= 2048) {
        layernorm<8, 256><<<nRow, 256, 0, stream>>>(nCol, epsilon, inp, gamma, beta, out, training, rstd);
    }
    else if (nCol <= 4096) {
        layernorm<8, 512><<<nRow, 512, 0, stream>>>(nCol, epsilon, inp, gamma, beta, out, training, rstd);
    }
}


template <int iterCnt, int stride>
__global__ void
biasedLayernorm(int nCol,
                float epsilon,
                const float* __restrict__ inp,
                const float* __restrict__ bias,
                const float* __restrict__ residual,
                const float* __restrict__ gamma,
                const float* __restrict__ beta,
                float* __restrict__ out,
                bool training,
                float* __restrict__ rstd) {
    constexpr int nWarp = stride >> 5;      // power of 2
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);

    inp += blockIdx.x * nCol;
    residual += blockIdx.x * nCol;
    out += blockIdx.x * nCol;

    float buff[iterCnt];
    float sumE = 0.f;
    float sumV = 0.f;
    for (int it = 0; it < iterCnt; ++it) {
        int idx = threadIdx.x + it * stride;
        if (idx >= nCol) break;
        sumE += buff[it] = inp[idx] + bias[idx] + residual[idx];
        sumV += square(buff[it]);
    }
    for (int i = 1; i < 32; i <<= 1) {
        sumE += tile.shfl_xor(sumE, i);
        sumV += tile.shfl_xor(sumV, i);
    }

    __shared__ float smE[nWarp];
    __shared__ float smV[nWarp];
    if (tile.thread_rank() == 0) {
        smE[threadIdx.x >> 5] = sumE;
        smV[threadIdx.x >> 5] = sumV;
    }
    block.sync();

    if (tile.thread_rank() < nWarp) {
        sumE = smE[tile.thread_rank()];
        sumV = smV[tile.thread_rank()];
    }
    for (int i = 1; i < nWarp; i <<= 1) {
        sumE += tile.shfl_xor(sumE, i);
        sumV += tile.shfl_xor(sumV, i);
    }
    sumE = tile.shfl(sumE, 0);
    sumV = tile.shfl(sumV, 0);

    float meanF = sumE / nCol;
    float rstdF = rsqrtf(sumV / nCol - square(meanF) + epsilon);

    for (int it = 0; it < iterCnt; ++it) {
        int idx = threadIdx.x + it * stride;
        if (idx >= nCol) break;
        out[idx] = (buff[it] - meanF) * rstdF * gamma[idx] + beta[idx];
    }
    if (training && threadIdx.x == 0) {
        rstd[blockIdx.x] = rstdF;
    }
}


template <int iterCnt, int stride>
__global__ void
biasedLayernorm(int nCol,
                float epsilon,
                const __half* __restrict__ inp,
                const __half* __restrict__ bias,
                const __half* __restrict__ residual,
                const __half* __restrict__ gamma,
                const __half* __restrict__ beta,
                __half* __restrict__ out,
                bool training,
                __half* __restrict__ rstd) {
    constexpr int nWarp = stride >> 5;      // power of 2
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);

    const __half2* inp2 = reinterpret_cast<const __half2*>(inp) + blockIdx.x * nCol;
    const __half2* bias2 = reinterpret_cast<const __half2*>(bias);
    const __half2* residual2 = reinterpret_cast<const __half2*>(residual) + blockIdx.x * nCol;
    const __half2* gamma2 = reinterpret_cast<const __half2*>(gamma);
    const __half2* beta2 = reinterpret_cast<const __half2*>(beta);
    __half2* out2 = reinterpret_cast<__half2*>(out) + blockIdx.x * nCol;

    float2 buff2[iterCnt];
    float sumE = 0.f;
    float sumV = 0.f;
    for (int it = 0; it < iterCnt; ++it) {
        int idx = threadIdx.x + it * stride;
        if (idx >= nCol) break;
        buff2[it] = __half22float2(inp2[idx] + bias2[idx] + residual2[idx]);
        sumE += buff2[it].x + buff2[it].y;
        sumV += square(buff2[it].x) + square(buff2[it].y);
    }
    for (int i = 1; i < 32; i <<= 1) {
        sumE += tile.shfl_xor(sumE, i);
        sumV += tile.shfl_xor(sumV, i);
    }

    __shared__ float smE[nWarp];
    __shared__ float smV[nWarp];
    if (tile.thread_rank() == 0) {
        smE[threadIdx.x >> 5] = sumE;
        smV[threadIdx.x >> 5] = sumV;
    }
    block.sync();

    if (tile.thread_rank() < nWarp) {
        sumE = smE[tile.thread_rank()];
        sumV = smV[tile.thread_rank()];
    }
    for (int i = 1; i < nWarp; i <<= 1) {
        sumE += tile.shfl_xor(sumE, i);
        sumV += tile.shfl_xor(sumV, i);
    }
    sumE = tile.shfl(sumE, 0);
    sumV = tile.shfl(sumV, 0);

    sumE /= nCol << 1;
    sumV /= nCol << 1;
    __half2 mean2 = __float2half2_rn(sumE);
    __half2 rstd2 = __float2half2_rn(rsqrtf(sumV - square(sumE) + epsilon));

    for (int it = 0; it < iterCnt; ++it) {
        int idx = threadIdx.x + it * stride;
        if (idx >= nCol) break;
        out2[idx] = (__float22half2_rn(buff2[it]) - mean2) * rstd2 * gamma2[idx] + beta2[idx];
    }
    if (training && threadIdx.x == 0) {
        rstd[blockIdx.x] = rstd2.x;
    }
}


// out-of-place biasedLayernorm forward
template <typename T>
inline void
launchBiasedLayernorm(int nRow,
                      int nCol,
                      float epsilon,
                      const T* inp,
                      const T* bias,
                      const T* residual,
                      const T* gamma,
                      const T* beta,
                      T* out,
                      bool training,
                      T* rstd = nullptr,
                      cudaStream_t stream = 0) {
    if (typeid(T) == typeid(__half)) {
        nCol >>= 1;
    }
    if (nCol <= 128) {
        biasedLayernorm<2, 64><<<nRow, 64, 0, stream>>>(nCol, epsilon, inp, bias, residual, gamma, beta, out, training, rstd);
    }
    else if (nCol <= 256) {
        biasedLayernorm<4, 64><<<nRow, 64, 0, stream>>>(nCol, epsilon, inp, bias, residual, gamma, beta, out, training, rstd);
    }
    else if (nCol <= 512) {
        biasedLayernorm<8, 64><<<nRow, 64, 0, stream>>>(nCol, epsilon, inp, bias, residual, gamma, beta, out, training, rstd);
    }
    else if (nCol <= 1024) {
        biasedLayernorm<8, 128><<<nRow, 128, 0, stream>>>(nCol, epsilon, inp, bias, residual, gamma, beta, out, training, rstd);
    }
    else if (nCol <= 2048) {
        biasedLayernorm<8, 256><<<nRow, 256, 0, stream>>>(nCol, epsilon, inp, bias, residual, gamma, beta, out, training, rstd);
    }
    else if (nCol <= 4096) {
        biasedLayernorm<8, 512><<<nRow, 512, 0, stream>>>(nCol, epsilon, inp, bias, residual, gamma, beta, out, training, rstd);
    }
}


template <int iterCnt, int stride>
__global__ void
layernormBackward(int nCol,
                  const float* __restrict__ grad,
                  const float* __restrict__ out,
                  const float* __restrict__ gamma,
                  const float* __restrict__ beta,
                  const float* __restrict__ rstd,
                  float* __restrict__ dInp) {
    constexpr int nWarp = stride >> 5;      // power of 2
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);

    grad += blockIdx.x * nCol;
    out += blockIdx.x * nCol;
    dInp += blockIdx.x * nCol;

    float ggam[iterCnt];
    float xhat[iterCnt];
    float sumE = 0.f;
    float sumV = 0.f;
    for (int it = 0; it < iterCnt; ++it) {
        int idx = it * stride + threadIdx.x;
        if (idx >= nCol) break;
        ggam[it] = grad[idx] * gamma[idx];
        xhat[it] = (out[idx] - beta[idx]) / gamma[idx];
        sumE += ggam[it];
        sumV += ggam[it] * xhat[it];
    }
    for (int i = 1; i < 32; i <<= 1) {
        sumE += tile.shfl_xor(sumE, i);
        sumV += tile.shfl_xor(sumV, i);
    }

    __shared__ float smE[nWarp];
    __shared__ float smV[nWarp];
    if (tile.thread_rank() == 0) {
        smE[threadIdx.x >> 5] = sumE;
        smV[threadIdx.x >> 5] = sumV;
    }
    block.sync();

    if (tile.thread_rank() < nWarp) {
        sumE = smE[tile.thread_rank()];
        sumV = smV[tile.thread_rank()];
    }
    for (int i = 1; i < nWarp; i <<= 1) {
        sumE += tile.shfl_xor(sumE, i);
        sumV += tile.shfl_xor(sumV, i);
    }
    sumE = tile.shfl(sumE, 0);
    sumV = tile.shfl(sumV, 0);

    sumE /= nCol;
    sumV /= nCol;

    float rstdF = rstd[blockIdx.x];
    for (int it = 0; it < iterCnt; ++it) {
        int idx = it * stride + threadIdx.x;
        if (idx >= nCol) break;
        dInp[idx] = rstdF * (ggam[it] - xhat[it] * sumV - sumE);
    }
}


template <int iterCnt, int stride>
__global__ void
layernormBackward(int nCol,
                  const __half* __restrict__ grad,
                  const __half* __restrict__ out,
                  const __half* __restrict__ gamma,
                  const __half* __restrict__ beta,
                  const __half* __restrict__ rstd,
                  __half* __restrict__ dInp) {
    constexpr int nWarp = stride >> 5;      // power of 2
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);

    const __half2* grad2 = reinterpret_cast<const __half2*>(grad) + blockIdx.x * nCol;
    const __half2* out2 = reinterpret_cast<const __half2*>(out) + blockIdx.x * nCol;
    const __half2* gamma2 = reinterpret_cast<const __half2*>(gamma);
    const __half2* beta2 = reinterpret_cast<const __half2*>(beta);
    __half2* dInp2 = reinterpret_cast<__half2*>(dInp) + blockIdx.x * nCol;

    __half2 ggam2[iterCnt];
    __half2 xhat2[iterCnt];
    float sumE = 0.f;
    float sumV = 0.f;
    for (int it = 0; it < iterCnt; ++it) {
        int idx = it * stride + threadIdx.x;
        if (idx >= nCol) break;
        ggam2[it] = grad2[idx] * gamma2[idx];
        xhat2[it] = (out2[idx] - beta2[idx]) / gamma2[idx];
        float2 tempE = __half22float2(ggam2[it]);
        float2 tempV = __half22float2(ggam2[it] * xhat2[it]);
        sumE += tempE.x + tempE.y;
        sumV += tempV.x + tempV.y;
    }
    for (int i = 1; i < 32; i <<= 1) {
        sumE += tile.shfl_xor(sumE, i);
        sumV += tile.shfl_xor(sumV, i);
    }

    __shared__ float smE[nWarp];
    __shared__ float smV[nWarp];
    if (tile.thread_rank() == 0) {
        smE[threadIdx.x >> 5] = sumE;
        smV[threadIdx.x >> 5] = sumV;
    }
    block.sync();

    if (tile.thread_rank() < nWarp) {
        sumE = smE[tile.thread_rank()];
        sumV = smV[tile.thread_rank()];
    }
    for (int i = 1; i < nWarp; i <<= 1) {
        sumE += tile.shfl_xor(sumE, i);
        sumV += tile.shfl_xor(sumV, i);
    }
    sumE = tile.shfl(sumE, 0);
    sumV = tile.shfl(sumV, 0);

    __half2 sumE2 = __float2half2_rn(sumE / (nCol << 1));
    __half2 sumV2 = __float2half2_rn(sumV / (nCol << 1));

    __half2 rstd2 = __half2half2(rstd[blockIdx.x]);
    for (int it = 0; it < iterCnt; ++it) {
        int idx = it * stride + threadIdx.x;
        if (idx >= nCol) break;
        dInp2[idx] = rstd2 * (ggam2[it] - xhat2[it] * sumV2 - sumE2);
    }
}


// out-of-place layernorm backward
template <typename T>
inline void
launchLayernormBackward(int nRow,
                        int nCol,
                        const T* grad,
                        const T* out,
                        const T* gamma,
                        const T* beta,
                        const T* rstd,
                        T* dInp,
                        cudaStream_t stream = 0) {
    if (typeid(T) == typeid(__half)) {
        nCol >>= 1;
    }
    if (nCol <= 128) {
        layernormBackward<2, 64><<<nRow, 64, 0, stream>>>(nCol, grad, out, gamma, beta, rstd, dInp);
    }
    else if (nCol <= 256) {
        layernormBackward<4, 64><<<nRow, 64, 0, stream>>>(nCol, grad, out, gamma, beta, rstd, dInp);
    }
    else if (nCol <= 512) {
        layernormBackward<8, 64><<<nRow, 64, 0, stream>>>(nCol, grad, out, gamma, beta, rstd, dInp);
    }
    else if (nCol <= 1024) {
        layernormBackward<8, 128><<<nRow, 128, 0, stream>>>(nCol, grad, out, gamma, beta, rstd, dInp);
    }
    else if (nCol <= 2048) {
        layernormBackward<8, 256><<<nRow, 256, 0, stream>>>(nCol, grad, out, gamma, beta, rstd, dInp);
    }
    else if (nCol <= 4096) {
        layernormBackward<8, 512><<<nRow, 512, 0, stream>>>(nCol, grad, out, gamma, beta, rstd, dInp);
    }
}


template <int iterCnt, int stride>
__global__ void
layernormBackwardAccum(int nCol,
                       const float* __restrict__ grad,
                       const float* __restrict__ out,
                       const float* __restrict__ gamma,
                       const float* __restrict__ beta,
                       const float* __restrict__ rstd,
                       float* __restrict__ dInp) {
    constexpr int nWarp = stride >> 5;      // power of 2
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);

    grad += blockIdx.x * nCol;
    out += blockIdx.x * nCol;
    dInp += blockIdx.x * nCol;

    float ggam[iterCnt];
    float xhat[iterCnt];
    float sumE = 0.f;
    float sumV = 0.f;
    for (int it = 0; it < iterCnt; ++it) {
        int idx = it * stride + threadIdx.x;
        if (idx >= nCol) break;
        ggam[it] = grad[idx] * gamma[idx];
        xhat[it] = (out[idx] - beta[idx]) / gamma[idx];
        sumE += ggam[it];
        sumV += ggam[it] * xhat[it];
    }
    for (int i = 1; i < 32; i <<= 1) {
        sumE += tile.shfl_xor(sumE, i);
        sumV += tile.shfl_xor(sumV, i);
    }

    __shared__ float smE[nWarp];
    __shared__ float smV[nWarp];
    if (tile.thread_rank() == 0) {
        smE[threadIdx.x >> 5] = sumE;
        smV[threadIdx.x >> 5] = sumV;
    }
    block.sync();

    if (tile.thread_rank() < nWarp) {
        sumE = smE[tile.thread_rank()];
        sumV = smV[tile.thread_rank()];
    }
    for (int i = 1; i < nWarp; i <<= 1) {
        sumE += tile.shfl_xor(sumE, i);
        sumV += tile.shfl_xor(sumV, i);
    }
    sumE = tile.shfl(sumE, 0);
    sumV = tile.shfl(sumV, 0);

    sumE /= nCol;
    sumV /= nCol;

    float rstdF = rstd[blockIdx.x];
    for (int it = 0; it < iterCnt; ++it) {
        int idx = it * stride + threadIdx.x;
        if (idx >= nCol) break;
        dInp[idx] += rstdF * (ggam[it] - xhat[it] * sumV - sumE);
    }
}


template <int iterCnt, int stride>
__global__ void
layernormBackwardAccum(int nCol,
                       const __half* __restrict__ grad,
                       const __half* __restrict__ out,
                       const __half* __restrict__ gamma,
                       const __half* __restrict__ beta,
                       const __half* __restrict__ rstd,
                       __half* __restrict__ dInp) {
    constexpr int nWarp = stride >> 5;      // power of 2
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);

    const __half2* grad2 = reinterpret_cast<const __half2*>(grad) + blockIdx.x * nCol;
    const __half2* out2 = reinterpret_cast<const __half2*>(out) + blockIdx.x * nCol;
    const __half2* gamma2 = reinterpret_cast<const __half2*>(gamma);
    const __half2* beta2 = reinterpret_cast<const __half2*>(beta);
    __half2* dInp2 = reinterpret_cast<__half2*>(dInp) + blockIdx.x * nCol;

    __half2 ggam2[iterCnt];
    __half2 xhat2[iterCnt];
    float sumE = 0.f;
    float sumV = 0.f;
    for (int it = 0; it < iterCnt; ++it) {
        int idx = it * stride + threadIdx.x;
        if (idx >= nCol) break;
        ggam2[it] = grad2[idx] * gamma2[idx];
        xhat2[it] = (out2[idx] - beta2[idx]) / gamma2[idx];
        float2 tempE = __half22float2(ggam2[it]);
        float2 tempV = __half22float2(ggam2[it] * xhat2[it]);
        sumE += tempE.x + tempE.y;
        sumV += tempV.x + tempV.y;
    }
    for (int i = 1; i < 32; i <<= 1) {
        sumE += tile.shfl_xor(sumE, i);
        sumV += tile.shfl_xor(sumV, i);
    }

    __shared__ float smE[nWarp];
    __shared__ float smV[nWarp];
    if (tile.thread_rank() == 0) {
        smE[threadIdx.x >> 5] = sumE;
        smV[threadIdx.x >> 5] = sumV;
    }
    block.sync();

    if (tile.thread_rank() < nWarp) {
        sumE = smE[tile.thread_rank()];
        sumV = smV[tile.thread_rank()];
    }
    for (int i = 1; i < nWarp; i <<= 1) {
        sumE += tile.shfl_xor(sumE, i);
        sumV += tile.shfl_xor(sumV, i);
    }
    sumE = tile.shfl(sumE, 0);
    sumV = tile.shfl(sumV, 0);

    __half2 sumE2 = __float2half2_rn(sumE / (nCol << 1));
    __half2 sumV2 = __float2half2_rn(sumV / (nCol << 1));

    __half2 rstd2 = __half2half2(rstd[blockIdx.x]);
    for (int it = 0; it < iterCnt; ++it) {
        int idx = it * stride + threadIdx.x;
        if (idx >= nCol) break;
        dInp2[idx] += rstd2 * (ggam2[it] - xhat2[it] * sumV2 - sumE2);
    }
}


// inplace layernorm backward accum
template <typename T>
inline void
launchLayernormBackwardAccum(int nRow,
                             int nCol,
                             const T* grad,
                             const T* out,
                             const T* gamma,
                             const T* beta,
                             const T* rstd,
                             T* dInp,
                             cudaStream_t stream = 0) {
    if (typeid(T) == typeid(__half)) {
        nCol >>= 1;
    }
    if (nCol <= 128) {
        layernormBackwardAccum<2, 64><<<nRow, 64, 0, stream>>>(nCol, grad, out, gamma, beta, rstd, dInp);
    }
    else if (nCol <= 256) {
        layernormBackwardAccum<4, 64><<<nRow, 64, 0, stream>>>(nCol, grad, out, gamma, beta, rstd, dInp);
    }
    else if (nCol <= 512) {
        layernormBackwardAccum<8, 64><<<nRow, 64, 0, stream>>>(nCol, grad, out, gamma, beta, rstd, dInp);
    }
    else if (nCol <= 1024) {
        layernormBackwardAccum<8, 128><<<nRow, 128, 0, stream>>>(nCol, grad, out, gamma, beta, rstd, dInp);
    }
    else if (nCol <= 2048) {
        layernormBackwardAccum<8, 256><<<nRow, 256, 0, stream>>>(nCol, grad, out, gamma, beta, rstd, dInp);
    }
    else if (nCol <= 4096) {
        layernormBackwardAccum<8, 512><<<nRow, 512, 0, stream>>>(nCol, grad, out, gamma, beta, rstd, dInp);
    }
}


template <int iterCnt, int stride>
__global__ void
layernormBackwardAccum(int nCol,
                       const float* __restrict__ grad,
                       const float* __restrict__ dResidual,
                       const float* __restrict__ out,
                       const float* __restrict__ gamma,
                       const float* __restrict__ beta,
                       const float* __restrict__ rstd,
                       float* __restrict__ dInp) {
    constexpr int nWarp = stride >> 5;      // power of 2
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);

    grad += blockIdx.x * nCol;
    dResidual += blockIdx.x * nCol;
    out += blockIdx.x * nCol;
    dInp += blockIdx.x * nCol;

    float ggam[iterCnt];
    float xhat[iterCnt];
    float sumE = 0.f;
    float sumV = 0.f;
    for (int it = 0; it < iterCnt; ++it) {
        int idx = it * stride + threadIdx.x;
        if (idx >= nCol) break;
        ggam[it] = grad[idx] * gamma[idx];
        xhat[it] = (out[idx] - beta[idx]) / gamma[idx];
        sumE += ggam[it];
        sumV += ggam[it] * xhat[it];
    }
    for (int i = 1; i < 32; i <<= 1) {
        sumE += tile.shfl_xor(sumE, i);
        sumV += tile.shfl_xor(sumV, i);
    }

    __shared__ float smE[nWarp];
    __shared__ float smV[nWarp];
    if (tile.thread_rank() == 0) {
        smE[threadIdx.x >> 5] = sumE;
        smV[threadIdx.x >> 5] = sumV;
    }
    block.sync();

    if (tile.thread_rank() < nWarp) {
        sumE = smE[tile.thread_rank()];
        sumV = smV[tile.thread_rank()];
    }
    for (int i = 1; i < nWarp; i <<= 1) {
        sumE += tile.shfl_xor(sumE, i);
        sumV += tile.shfl_xor(sumV, i);
    }
    sumE = tile.shfl(sumE, 0);
    sumV = tile.shfl(sumV, 0);

    sumE /= nCol;
    sumV /= nCol;

    float rstdF = rstd[blockIdx.x];
    for (int it = 0; it < iterCnt; ++it) {
        int idx = it * stride + threadIdx.x;
        if (idx >= nCol) break;
        dInp[idx] = rstdF * (ggam[it] - xhat[it] * sumV - sumE) + dResidual[idx];
    }
}


template <int iterCnt, int stride>
__global__ void
layernormBackwardAccum(int nCol,
                       const __half* __restrict__ grad,
                       const __half* __restrict__ dResidual,
                       const __half* __restrict__ out,
                       const __half* __restrict__ gamma,
                       const __half* __restrict__ beta,
                       const __half* __restrict__ rstd,
                       __half* __restrict__ dInp) {
    constexpr int nWarp = stride >> 5;      // power of 2
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);

    const __half2* grad2 = reinterpret_cast<const __half2*>(grad) + blockIdx.x * nCol;
    const __half2* dResidual2 = reinterpret_cast<const __half2*>(dResidual) + blockIdx.x * nCol;
    const __half2* out2 = reinterpret_cast<const __half2*>(out) + blockIdx.x * nCol;
    const __half2* gamma2 = reinterpret_cast<const __half2*>(gamma);
    const __half2* beta2 = reinterpret_cast<const __half2*>(beta);
    __half2* dInp2 = reinterpret_cast<__half2*>(dInp) + blockIdx.x * nCol;

    __half2 ggam2[iterCnt];
    __half2 xhat2[iterCnt];
    float sumE = 0.f;
    float sumV = 0.f;
    for (int it = 0; it < iterCnt; ++it) {
        int idx = it * stride + threadIdx.x;
        if (idx >= nCol) break;
        ggam2[it] = grad2[idx] * gamma2[idx];
        xhat2[it] = (out2[idx] - beta2[idx]) / gamma2[idx];
        float2 tempE = __half22float2(ggam2[it]);
        float2 tempV = __half22float2(ggam2[it] * xhat2[it]);
        sumE += tempE.x + tempE.y;
        sumV += tempV.x + tempV.y;
    }
    for (int i = 1; i < 32; i <<= 1) {
        sumE += tile.shfl_xor(sumE, i);
        sumV += tile.shfl_xor(sumV, i);
    }

    __shared__ float smE[nWarp];
    __shared__ float smV[nWarp];
    if (tile.thread_rank() == 0) {
        smE[threadIdx.x >> 5] = sumE;
        smV[threadIdx.x >> 5] = sumV;
    }
    block.sync();

    if (tile.thread_rank() < nWarp) {
        sumE = smE[tile.thread_rank()];
        sumV = smV[tile.thread_rank()];
    }
    for (int i = 1; i < nWarp; i <<= 1) {
        sumE += tile.shfl_xor(sumE, i);
        sumV += tile.shfl_xor(sumV, i);
    }
    sumE = tile.shfl(sumE, 0);
    sumV = tile.shfl(sumV, 0);

    __half2 sumE2 = __float2half2_rn(sumE / (nCol << 1));
    __half2 sumV2 = __float2half2_rn(sumV / (nCol << 1));

    __half2 rstd2 = __half2half2(rstd[blockIdx.x]);
    for (int it = 0; it < iterCnt; ++it) {
        int idx = it * stride + threadIdx.x;
        if (idx >= nCol) break;
        dInp2[idx] = rstd2 * (ggam2[it] - xhat2[it] * sumV2 - sumE2) + dResidual2[idx];
    }
}


// out-of-place layernorm backward accum
template <typename T>
inline void
launchLayernormBackwardAccum(int nRow,
                             int nCol,
                             const T* grad,
                             const T* dResidual,
                             const T* out,
                             const T* gamma,
                             const T* beta,
                             const T* rstd,
                             T* dInp,
                             cudaStream_t stream = 0) {
    if (typeid(T) == typeid(__half)) {
        nCol >>= 1;
    }
    if (nCol <= 128) {
        layernormBackwardAccum<2, 64><<<nRow, 64, 0, stream>>>(nCol, grad, dResidual, out, gamma, beta, rstd, dInp);
    }
    else if (nCol <= 256) {
        layernormBackwardAccum<4, 64><<<nRow, 64, 0, stream>>>(nCol, grad, dResidual, out, gamma, beta, rstd, dInp);
    }
    else if (nCol <= 512) {
        layernormBackwardAccum<8, 64><<<nRow, 64, 0, stream>>>(nCol, grad, dResidual, out, gamma, beta, rstd, dInp);
    }
    else if (nCol <= 1024) {
        layernormBackwardAccum<8, 128><<<nRow, 128, 0, stream>>>(nCol, grad, dResidual, out, gamma, beta, rstd, dInp);
    }
    else if (nCol <= 2048) {
        layernormBackwardAccum<8, 256><<<nRow, 256, 0, stream>>>(nCol, grad, dResidual, out, gamma, beta, rstd, dInp);
    }
    else if (nCol <= 4096) {
        layernormBackwardAccum<8, 512><<<nRow, 512, 0, stream>>>(nCol, grad, dResidual, out, gamma, beta, rstd, dInp);
    }
}


template <typename T, int tileSize>
__global__ void
layernormBackwardGB(int nRow,
                    int nCol,
                    const T* __restrict__ grad,
                    const T* __restrict__ out,
                    const T* __restrict__ gamma,
                    const T* __restrict__ beta,
                    T* __restrict__ dGamma,
                    T* __restrict__ dBeta) {
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<tileSize>(block);

    __shared__ float smGamma[tileSize][tileSize + 1];
    __shared__ float smBeta[tileSize][tileSize + 1];
    float sumGamma = 0.f;
    float sumBeta = 0.f;

    int x = blockIdx.x * tileSize + threadIdx.x;
    if (x < nCol) {
        int offset = threadIdx.y * nCol + x;
        int stride = tileSize * nCol;

        for (int y = threadIdx.y; y < nRow; y += tileSize) {
            float tpGrad = float(grad[offset]);
            float xhat = (float(out[offset]) - float(beta[x])) / float(gamma[x]);
            sumGamma += tpGrad * xhat;
            sumBeta += tpGrad;
            offset += stride;
        }
    }

    smGamma[threadIdx.y][threadIdx.x] = sumGamma;
    smBeta[threadIdx.y][threadIdx.x] = sumBeta;
    block.sync();

    sumGamma = smGamma[threadIdx.x][threadIdx.y];
    sumBeta = smBeta[threadIdx.x][threadIdx.y];
    for (int i = 1; i < tileSize; i <<= 1) {
        sumGamma += tile.shfl_xor(sumGamma, i);
        sumBeta += tile.shfl_xor(sumBeta, i);
    }

    if (threadIdx.x == 0) {
        int x = blockIdx.x * tileSize + threadIdx.y;
        if (x < nCol) {
            dGamma[x] = sumGamma;
            dBeta[x] = sumBeta;
        }
    }
}


template <typename T, int tileSize>
__global__ void
biasedLayernormBackwardGB(int nRow,
                          int nCol,
                          const T* __restrict__ grad,
                          const T* __restrict__ out,
                          const T* __restrict__ gamma,
                          const T* __restrict__ beta,
                          const T* __restrict__ dInp,
                          T* __restrict__ dGamma,
                          T* __restrict__ dBeta,
                          T* __restrict__ dBias,
                          T* __restrict__ dResidual) {
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<tileSize>(block);

    __shared__ float smGamma[tileSize][tileSize + 1];
    __shared__ float smBeta[tileSize][tileSize + 1];
    __shared__ float smBias[tileSize][tileSize + 1];
    float sumGamma = 0.f;
    float sumBeta = 0.f;
    float sumBias = 0.f;

    int x = blockIdx.x * tileSize + threadIdx.x;
    if (x < nCol) {
        int offset = threadIdx.y * nCol + x;
        int stride = tileSize * nCol;

        for (int y = threadIdx.y; y < nRow; y += tileSize) {
            float xhat = (float(out[offset]) - float(beta[x])) / float(gamma[x]);
            sumGamma += float(grad[offset]) * xhat;
            sumBeta += float(grad[offset]);
            sumBias += float(dInp[offset]);
            dResidual[offset] = dInp[offset];
            offset += stride;
        }
    }

    smGamma[threadIdx.y][threadIdx.x] = sumGamma;
    smBeta[threadIdx.y][threadIdx.x] = sumBeta;
    smBias[threadIdx.y][threadIdx.x] = sumBias;
    block.sync();

    sumGamma = smGamma[threadIdx.x][threadIdx.y];
    sumBeta = smBeta[threadIdx.x][threadIdx.y];
    sumBias = smBias[threadIdx.x][threadIdx.y];
    for (int i = 1; i < tileSize; i <<= 1) {
        sumGamma += tile.shfl_xor(sumGamma, i);
        sumBeta += tile.shfl_xor(sumBeta, i);
        sumBias += tile.shfl_xor(sumBias, i);
    }

    if (threadIdx.x == 0) {
        int x = blockIdx.x * tileSize + threadIdx.y;
        if (x < nCol) {
            dGamma[x] = sumGamma;
            dBeta[x] = sumBeta;
            dBias[x] = sumBias;
        }
    }
}


template <typename T>
class LayerNorm {
public:
    static void
    forward(const T* inp,
            const T* gamma,
            const T* beta,
            T* out,
            T* rstd,
            bool training,
            float eps,
            int batchCnt,
            int hidden_size,
            cudaStream_t stream = 0) {
#ifdef _DEBUG
        if (typeid(T) == typeid(__half)) {
            if (hidden_size & 1 || hidden_size > 8192) {
                errMsg(format("Unsupported hidden_size (%d) for Layernorm with FP16", hidden_size));
            }
        }
        else if (hidden_size > 4096) {
            errMsg(format("Unsupported hidden_size (%d) for Layernorm", hidden_size));
        }
#endif
        launchLayernorm(batchCnt,
                        hidden_size,
                        eps,
                        inp,
                        gamma,
                        beta,
                        out,
                        training,
                        rstd,
                        stream);
    }

    static void
    forwardB(const T* inp,
             const T* bias,
             const T* residual,
             const T* gamma,
             const T* beta,
             T* out,
             T* rstd,
             bool training,
             float eps,
             int batchCnt,
             int hidden_size,
             cudaStream_t stream = 0) {
#ifdef _DEBUG
        if (typeid(T) == typeid(__half)) {
            if (hidden_size & 1 || hidden_size > 8192) {
                errMsg(format("Unsupported hidden_size (%d) for Layernorm with FP16", hidden_size));
            }
        }
        else if (hidden_size > 4096) {
            errMsg(format("Unsupported hidden_size (%d) for Layernorm", hidden_size));
        }
#endif
        launchBiasedLayernorm(batchCnt,
                              hidden_size,
                              eps,
                              inp,
                              bias,
                              residual,
                              gamma,
                              beta,
                              out,
                              training,
                              rstd,
                              stream);
    }

    static void
    backward(const T* grad,
             const T* out,
             const T* gamma,
             const T* beta,
             const T* rstd,
             T* dGamma,
             T* dBeta,
             T* dInp,
             int batchCnt,
             int hidden_size,
             cudaStream_t stream = 0) {
        launchLayernormBackward(batchCnt,
                                hidden_size,
                                grad,
                                out,
                                gamma,
                                beta,
                                rstd,
                                dInp,
                                stream);

        layernormBackwardGB<T, 32><<<ceilDiv(hidden_size, 32), dim3(32, 32), 0, stream>>>(batchCnt, hidden_size, grad, out, gamma, beta, dGamma, dBeta);
    }

    static void
    backwardA(const T* grad,
              const T* out,
              const T* gamma,
              const T* beta,
              const T* rstd,
              T* dGamma,
              T* dBeta,
              T* dInp,
              int batchCnt,
              int hidden_size,
              cudaStream_t stream = 0) {
        launchLayernormBackwardAccum(batchCnt,
                                     hidden_size,
                                     grad,
                                     out,
                                     gamma,
                                     beta,
                                     rstd,
                                     dInp,
                                     stream);

        layernormBackwardGB<T, 32><<<ceilDiv(hidden_size, 32), dim3(32, 32), 0, stream>>>(batchCnt, hidden_size, grad, out, gamma, beta, dGamma, dBeta);
    }

    static void
    backwardA(const T* grad,
              const T* dResidual,
              const T* out,
              const T* gamma,
              const T* beta,
              const T* rstd,
              T* dGamma,
              T* dBeta,
              T* dInp,
              int batchCnt,
              int hidden_size,
              cudaStream_t stream = 0) {
        launchLayernormBackwardAccum(batchCnt,
                                     hidden_size,
                                     grad,
                                     dResidual,
                                     out,
                                     gamma,
                                     beta,
                                     rstd,
                                     dInp,
                                     stream);

        layernormBackwardGB<T, 32><<<ceilDiv(hidden_size, 32), dim3(32, 32), 0, stream>>>(batchCnt, hidden_size, grad, out, gamma, beta, dGamma, dBeta);
    }

    static void
    backwardB(const T* grad,
              const T* out,
              const T* gamma,
              const T* beta,
              const T* rstd,
              T* dGamma,
              T* dBeta,
              T* dInp,
              T* dBias,
              T* dResidual,
              int batchCnt,
              int hidden_size,
              cudaStream_t stream = 0) {
        launchLayernormBackward(batchCnt,
                                hidden_size,
                                grad,
                                out,
                                gamma,
                                beta,
                                rstd,
                                dInp,
                                stream);

        biasedLayernormBackwardGB<T, 32><<<ceilDiv(hidden_size, 32), dim3(32, 32), 0, stream>>>(batchCnt, hidden_size, grad, out, gamma, beta, dInp, dGamma, dBeta, dBias, dResidual);
    }
};

