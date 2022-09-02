#include "Add.h"
#include "utils.h"
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <typeinfo>
namespace cg = cooperative_groups;


__global__ void
addBiasIF(int nCol,
          const float* __restrict__ bias,
          float* __restrict__ io) {
    const float4* bias4 = reinterpret_cast<const float4*>(bias);
    float4* io4 = reinterpret_cast<float4*>(io) + blockIdx.x * nCol;

    for (int i = threadIdx.x; i < nCol; i += blockDim.x) {
        io4[i].x += bias4[i].x;
        io4[i].y += bias4[i].y;
        io4[i].z += bias4[i].z;
        io4[i].w += bias4[i].w;
    }
}


__global__ void
addBiasIF(int nCol,
          const __half* __restrict__ bias,
          __half* __restrict__ io) {
    const float4* bias4 = reinterpret_cast<const float4*>(bias);
    float4* io4 = reinterpret_cast<float4*>(io) + blockIdx.x * nCol;

    __half2 i2[4], b2[4];
    for (int i = threadIdx.x; i < nCol; i += blockDim.x) {
        *reinterpret_cast<float4*>(i2) = io4[i];
        *reinterpret_cast<float4*>(b2) = bias4[i];
        i2[0] += b2[0];
        i2[1] += b2[1];
        i2[2] += b2[2];
        i2[3] += b2[3];
        io4[i] = *reinterpret_cast<float4*>(i2);
    }
}


template <typename T>
void
launchAddBiasIF(int nRow,
                int nCol,
                const T* bias,
                T* io,
                cudaStream_t stream) {
    int s = 2 + (typeid(T) == typeid(__half));
    nCol >>= s;
    const int dimBlock = min(1024, nCol);
    addBiasIF<<<nRow, dimBlock, 0, stream>>>(nCol, bias, io);
}


template
void
launchAddBiasIF(int nRow,
                int nCol,
                const float* bias,
                float* io,
                cudaStream_t stream);


template
void
launchAddBiasIF(int nRow,
                int nCol,
                const __half* bias,
                __half* io,
                cudaStream_t stream);


template <typename T, int tileSize>
__global__ void
addBiasOB(int nRow,
          int nCol,
          const T* __restrict__ grad,
          T* __restrict__ dBias) {
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<tileSize>(block);

    __shared__ float sm[tileSize][tileSize + 1];
    float sum = 0.f;

    int x = blockIdx.x * tileSize + threadIdx.x;
    if (x < nCol) {
        int offset = threadIdx.y * nCol + x;
        int stride = tileSize * nCol;
        for (int y = threadIdx.y; y < nRow; y += tileSize) {
            sum += float(__ldg(&grad[offset]));
            offset += stride;
        }
    }

    sm[threadIdx.y][threadIdx.x] = sum;
    block.sync();

    sum = sm[threadIdx.x][threadIdx.y];
    for (int i = 1; i < tileSize; i <<= 1) {
        sum += tile.shfl_xor(sum, i);
    }

    if (threadIdx.x == 0) {
        int x = blockIdx.x * tileSize + threadIdx.y;
        if (x < nCol) {
            dBias[x] = sum;
        }
    }
}


template <typename T>
void
launchAddBiasOB(int nRow,
                int nCol,
                const T* grad,
                T* dBias,
                cudaStream_t stream) {
    constexpr dim3 dimBlock(32, 32);
    int dimGrid = ceilDiv(nCol, 32);
    addBiasOB<T, 32><<<dimGrid, dimBlock, 0, stream>>>(nRow, nCol, grad, dBias);
}


template
void
launchAddBiasOB(int nRow,
                int nCol,
                const float* grad,
                float* dBias,
                cudaStream_t stream);


template
void
launchAddBiasOB(int nRow,
                int nCol,
                const __half* grad,
                __half* dBias,
                cudaStream_t stream);


__global__ void
addBiasResOF(int nCol,
             const float* __restrict__ inp,
             const float* __restrict__ bias,
             const float* __restrict__ residual,
             float* __restrict__ out) {
    const float4* inp4 = reinterpret_cast<const float4*>(inp) + blockIdx.x * nCol;
    const float4* bias4 = reinterpret_cast<const float4*>(bias);
    const float4* residual4 = reinterpret_cast<const float4*>(residual) + blockIdx.x * nCol;
    float4* out4 = reinterpret_cast<float4*>(out) + blockIdx.x * nCol;

    for (int i = threadIdx.x; i < nCol; i += blockDim.x) {
        out4[i].x = inp4[i].x + bias4[i].x + residual4[i].x;
        out4[i].y = inp4[i].y + bias4[i].y + residual4[i].y;
        out4[i].z = inp4[i].z + bias4[i].z + residual4[i].z;
        out4[i].w = inp4[i].w + bias4[i].w + residual4[i].w;
    }
}


__global__ void
addBiasResOF(int nCol,
             const __half* __restrict__ inp,
             const __half* __restrict__ bias,
             const __half* __restrict__ residual,
             __half* __restrict__ out) {
    const float4* inp4 = reinterpret_cast<const float4*>(inp) + blockIdx.x * nCol;
    const float4* bias4 = reinterpret_cast<const float4*>(bias);
    const float4* residual4 = reinterpret_cast<const float4*>(residual) + blockIdx.x * nCol;
    float4* out4 = reinterpret_cast<float4*>(out) + blockIdx.x * nCol;

    __half2 i2[4], b2[4], r2[4];
    for (int i = threadIdx.x; i < nCol; i += blockDim.x) {
        *reinterpret_cast<float4*>(i2) = inp4[i];
        *reinterpret_cast<float4*>(b2) = bias4[i];
        *reinterpret_cast<float4*>(r2) = residual4[i];
        r2[0] += i2[0] + b2[0];
        r2[1] += i2[1] + b2[1];
        r2[2] += i2[2] + b2[2];
        r2[3] += i2[3] + b2[3];
        out4[i] = *reinterpret_cast<float4*>(r2);
    }
}


template <typename T>
void
launchAddBiasResOF(int nRow,
                   int nCol,
                   const T* inp,
                   const T* bias,
                   const T* residual,
                   T* out,
                   cudaStream_t stream) {
    int s = 2 + (typeid(T) == typeid(__half));
    nCol >>= s;
    const int dimBlock = min(1024, nCol);
    addBiasResOF<<<nRow, dimBlock, 0, stream>>>(nCol, inp, bias, residual, out);
}


template
void
launchAddBiasResOF(int nRow,
                   int nCol,
                   const float* inp,
                   const float* bias,
                   const float* residual,
                   float* out,
                   cudaStream_t stream);


template
void
launchAddBiasResOF(int nRow,
                   int nCol,
                   const __half* inp,
                   const __half* bias,
                   const __half* residual,
                   __half* out,
                   cudaStream_t stream);


template <typename T, int tileSize>
__global__ void
addBiasResOB(int nRow,
             int nCol,
             const T* __restrict__ grad,
             T* __restrict__ dInp,
             T* __restrict__ dBias) {
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<tileSize>(block);

    __shared__ float sm[tileSize][tileSize + 1];
    float sum = 0.f;

    int x = blockIdx.x * tileSize + threadIdx.x;
    if (x < nCol) {
        int offset = threadIdx.y * nCol + x;
        int stride = tileSize * nCol;
        for (int y = threadIdx.y; y < nRow; y += tileSize) {
            dInp[offset] = __ldg(&grad[offset]);
            sum += float(__ldg(&grad[offset]));
            offset += stride;
        }
    }

    sm[threadIdx.y][threadIdx.x] = sum;
    block.sync();

    sum = sm[threadIdx.x][threadIdx.y];
    for (int i = 1; i < tileSize; i <<= 1) {
        sum += tile.shfl_xor(sum, i);
    }

    if (threadIdx.x == 0) {
        int x = blockIdx.x * tileSize + threadIdx.y;
        if (x < nCol) {
            dBias[x] = sum;
        }
    }
}


template <typename T>
void
launchAddBiasResOB(int nRow,
                   int nCol,
                   const T* grad,
                   T* dInp,
                   T* dBias,
                   cudaStream_t stream) {
    constexpr dim3 dimBlock(32, 32);
    int dimGrid = ceilDiv(nCol, 32);
    addBiasResOB<T, 32><<<dimGrid, dimBlock, 0, stream>>>(nRow, nCol, grad, dInp, dBias);
}


template
void
launchAddBiasResOB(int nRow,
                   int nCol,
                   const float* grad,
                   float* dInp,
                   float* dBias,
                   cudaStream_t stream);


template
void
launchAddBiasResOB(int nRow,
                   int nCol,
                   const __half* grad,
                   __half* dInp,
                   __half* dBias,
                   cudaStream_t stream);

