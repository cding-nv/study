#include "Gelu.h"
#include <cuda_fp16.h>
#include <typeinfo>
#define SQRT_2_DIV_PI 0.7978845608028654f


__device__ __forceinline__ float
geluForwardFunc(float x) {
    constexpr float param = SQRT_2_DIV_PI * 0.044715f;
    return x * 0.5f * (1.f + tanhf(x * (SQRT_2_DIV_PI + param * x * x)));
}


__global__ void
geluBiasIF(int nCol,
           const float* __restrict__ bias,
           float* __restrict__ io) {
    const float4* bias4 = reinterpret_cast<const float4*>(bias);
    float4* io4 = reinterpret_cast<float4*>(io) + blockIdx.x * nCol;

    float4 i4, b4;
    for (int i = threadIdx.x; i < nCol; i += blockDim.x) {
        i4 = io4[i];
        b4 = bias4[i];
        i4.x = geluForwardFunc(i4.x + b4.x);
        i4.y = geluForwardFunc(i4.y + b4.y);
        i4.z = geluForwardFunc(i4.z + b4.z);
        i4.w = geluForwardFunc(i4.w + b4.w);
        io4[i] = i4;
    }
}


__global__ void
geluBiasIF(int nCol,
           const __half* __restrict__ bias,
           __half* __restrict__ io) {
    const float4* bias4 = reinterpret_cast<const float4*>(bias);
    float4* io4 = reinterpret_cast<float4*>(io) + blockIdx.x * nCol;

    float4 i4, b4;
    __half2* i2 = reinterpret_cast<__half2*>(&i4);
    __half2* b2 = reinterpret_cast<__half2*>(&b4);
    float2 tp[4];
    for (int i = threadIdx.x; i < nCol; i += blockDim.x) {
        i4 = io4[i];
        b4 = bias4[i];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            tp[j] = __half22float2(i2[j] + b2[j]);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            tp[j].x = geluForwardFunc(tp[j].x);
            tp[j].y = geluForwardFunc(tp[j].y);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            i2[j] = __float22half2_rn(tp[j]);
        }
        io4[i] = i4;
    }
}


template <typename T>
void
launchGeluBiasIF(int nRow,
                 int nCol,
                 const T* bias,
                 T* io,
                 cudaStream_t stream) {
    int s = 2 + (typeid(T) == typeid(__half));
    nCol >>= s;
    const int dimBlock = min(1024, nCol);
    geluBiasIF<<<nRow, dimBlock, 0, stream>>>(nCol, bias, io);
}


template
void
launchGeluBiasIF(int nRow,
                 int nCol,
                 const float* bias,
                 float* io,
                 cudaStream_t stream);


template
void
launchGeluBiasIF(int nRow,
                 int nCol,
                 const __half* bias,
                 __half* io,
                 cudaStream_t stream);


__global__ void
geluBiasOF(int nCol,
           const float* __restrict__ inp,
           const float* __restrict__ bias,
           float* __restrict__ out) {
    const float4* inp4 = reinterpret_cast<const float4*>(inp) + blockIdx.x * nCol;
    const float4* bias4 = reinterpret_cast<const float4*>(bias);
    float4* out4 = reinterpret_cast<float4*>(out) + blockIdx.x * nCol;

    float4 i4, b4;
    for (int i = threadIdx.x; i < nCol; i += blockDim.x) {
        i4 = inp4[i];
        b4 = bias4[i];
        i4.x = geluForwardFunc(i4.x + b4.x);
        i4.y = geluForwardFunc(i4.y + b4.y);
        i4.z = geluForwardFunc(i4.z + b4.z);
        i4.w = geluForwardFunc(i4.w + b4.w);
        out4[i] = i4;
    }
}


__global__ void
geluBiasOF(int nCol,
           const __half* __restrict__ inp,
           const __half* __restrict__ bias,
           __half* __restrict__ out) {
    const float4* inp4 = reinterpret_cast<const float4*>(inp) + blockIdx.x * nCol;
    const float4* bias4 = reinterpret_cast<const float4*>(bias);
    float4* out4 = reinterpret_cast<float4*>(out) + blockIdx.x * nCol;

    float4 i4, b4;
    __half2* i2 = reinterpret_cast<__half2*>(&i4);
    __half2* b2 = reinterpret_cast<__half2*>(&b4);
    float2 tp[4];
    for (int i = threadIdx.x; i < nCol; i += blockDim.x) {
        i4 = inp4[i];
        b4 = bias4[i];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            tp[j] = __half22float2(i2[j] + b2[j]);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            tp[j].x = geluForwardFunc(tp[j].x);
            tp[j].y = geluForwardFunc(tp[j].y);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            i2[j] = __float22half2_rn(tp[j]);
        }
        out4[i] = i4;
    }
}


template <typename T>
void
launchGeluBiasOF(int nRow,
                 int nCol,
                 const T* inp,
                 const T* bias,
                 T* out,
                 cudaStream_t stream) {
    int s = 2 + (typeid(T) == typeid(__half));
    nCol >>= s;
    const int dimBlock = min(1024, nCol);
    geluBiasOF<<<nRow, dimBlock, 0, stream>>>(nCol, inp, bias, out);
}


template
void
launchGeluBiasOF(int nRow,
                 int nCol,
                 const float* inp,
                 const float* bias,
                 float* out,
                 cudaStream_t stream);


template
void
launchGeluBiasOF(int nRow,
                 int nCol,
                 const __half* inp,
                 const __half* bias,
                 __half* out,
                 cudaStream_t stream);


__device__ __forceinline__ float
geluBackwardFunc(float x) {
    float tmpS = 0.044715f * x * x;
    float tmpX = SQRT_2_DIV_PI * x;
    float tmpTanh = tanhf(tmpX * tmpS + tmpX);
    return 0.5f * (tmpTanh + tmpX * (1.f - tmpTanh * tmpTanh) * (1.f + 3.f * tmpS)) + 0.5f;
}


__global__ void
geluBiasIB(int nCol,
           float* __restrict__ grad,
           const float* __restrict__ inp,
           const float* __restrict__ bias) {
    float4* grad4 = reinterpret_cast<float4*>(grad) + blockIdx.x * nCol;
    const float4* inp4 = reinterpret_cast<const float4*>(inp) + blockIdx.x * nCol;
    const float4* bias4 = reinterpret_cast<const float4*>(bias);

    float4 g4, i4, b4;
    for (int i = threadIdx.x; i < nCol; i += blockDim.x) {
        g4 = grad4[i];
        i4 = inp4[i];
        b4 = bias4[i];
        g4.x *= geluBackwardFunc(i4.x + b4.x);
        g4.y *= geluBackwardFunc(i4.y + b4.y);
        g4.z *= geluBackwardFunc(i4.z + b4.z);
        g4.w *= geluBackwardFunc(i4.w + b4.w);
        grad4[i] = g4;
    }
}


__global__ void
geluBiasIB(int nCol,
           __half* __restrict__ grad,
           const __half* __restrict__ inp,
           const __half* __restrict__ bias) {
    float4* grad4 = reinterpret_cast<float4*>(grad) + blockIdx.x * nCol;
    const float4* inp4 = reinterpret_cast<const float4*>(inp) + blockIdx.x * nCol;
    const float4* bias4 = reinterpret_cast<const float4*>(bias);

    float4 g4, i4, b4;
    __half2* g2 = reinterpret_cast<__half2*>(&g4);
    __half2* i2 = reinterpret_cast<__half2*>(&i4);
    __half2* b2 = reinterpret_cast<__half2*>(&b4);
    float2 tp[4];
    for (int i = threadIdx.x; i < nCol; i += blockDim.x) {
        g4 = grad4[i];
        i4 = inp4[i];
        b4 = bias4[i];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            tp[j] = __half22float2(i2[j] + b2[j]);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            tp[j].x = geluBackwardFunc(tp[j].x);
            tp[j].y = geluBackwardFunc(tp[j].y);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            g2[j] *= __float22half2_rn(tp[j]);
        }
        grad4[i] = g4;
    }
}


template <typename T>
void
launchGeluBiasIB(int nRow,
                 int nCol,
                 T* grad,
                 const T* inp,
                 const T* bias,
                 cudaStream_t stream) {
    int s = 2 + (typeid(T) == typeid(__half));
    nCol >>= s;
    const int dimBlock = min(1024, nCol);
    geluBiasIB<<<nRow, dimBlock, 0, stream>>>(nCol, grad, inp, bias);
}


template
void
launchGeluBiasIB(int nRow,
                 int nCol,
                 float* grad,
                 const float* inp,
                 const float* bias,
                 cudaStream_t stream);


template
void
launchGeluBiasIB(int nRow,
                 int nCol,
                 __half* grad,
                 const __half* inp,
                 const __half* bias,
                 cudaStream_t stream);

