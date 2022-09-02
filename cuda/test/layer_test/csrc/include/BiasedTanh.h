#pragma once
#include "Add.h"
#include "errMsg.h"
#include "utils.h"
#include <cuda_fp16.h>
#include <typeinfo>


__device__ __forceinline__ float
tanhForwardFunc(float x) {
    return tanhf(x);
}


__device__ __forceinline__ __half2
tanhForwardFunc(__half2 x2) {
    float2 temp2 = __half22float2(x2);
    temp2.x = tanhf(temp2.x);
    temp2.y = tanhf(temp2.y);
    return __float22half2_rn(temp2);
}


template <typename T>
__global__ void
biasedTanh(int nCol,
           const T* __restrict__ bias,
           T* __restrict__ io) {
    io += blockIdx.x * nCol;

    for (int i = threadIdx.x; i < nCol; i += blockDim.x) {
        io[i] = tanhForwardFunc(io[i] + bias[i]);
    }
}


// inplace biasedTanh forward
template <typename T>
inline void
launchBiasedTanh(int nRow,
                 int nCol,
                 const T* bias,
                 T* io,
                 cudaStream_t stream = 0) {
    const int dimBlock = min(1024, nCol >> 2);
    biasedTanh<<<nRow, dimBlock, 0, stream>>>(nCol, bias, io);
}


__device__ __forceinline__ float
tanhBackwardFunc(float x) {
    return 1.f - x * x;
}


__device__ __forceinline__ __half2
tanhBackwardFunc(__half2 x2) {
    float2 temp2 = __half22float2(x2);
    temp2.x = tanhBackwardFunc(temp2.x);
    temp2.y = tanhBackwardFunc(temp2.y);
    return __float22half2_rn(temp2);
}


template <typename T>
__global__ void
biasedTanhBackward(int nCol,
                   const T* __restrict__ grad,
                   const T* __restrict__ out,
                   T* __restrict__ dInp) {
    grad += blockIdx.x * nCol;
    out += blockIdx.x * nCol;
    dInp += blockIdx.x * nCol;

    for (int i = threadIdx.x; i < nCol; i += blockDim.x) {
        dInp[i] = grad[i] * T(tanhBackwardFunc(out[i]));
    }
}


// out-of-place biasedTanh backward
template <typename T>
inline void
launchBiasedTanhBackward(int nRow,
                         int nCol,
                         const T* grad,
                         const T* out,
                         T* dInp,
                         cudaStream_t stream = 0) {
    const int dimBlock = min(1024, nCol >> 2);
    biasedTanhBackward<<<nRow, dimBlock, 0, stream>>>(nCol, grad, out, dInp);
}


template <typename T>
class BiasedTanh {
public:
    static void
    forward(const T* bias,
            T* io,
            int nRow,
            int nCol,
            cudaStream_t stream = 0) {
        if (typeid(T) == typeid(__half)) {
#ifdef _DEBUG
            if (nCol & 1) {
                errMsg(format("Unsupported size (%d) for BiasedTanh with FP16", nCol));
            }
#endif
            launchBiasedTanh(nRow,
                             nCol >> 1,
                             reinterpret_cast<const __half2*>(bias),
                             reinterpret_cast<__half2*>(io),
                             stream);
        }
        else {
            launchBiasedTanh(nRow,
                             nCol,
                             bias,
                             io,
                             stream);
        }
    }

    static void
    backward(const T* grad,
             const T* out,
             T* dInp,
             T* dBias,
             int nRow,
             int nCol,
             cudaStream_t stream = 0) {
        if (typeid(T) == typeid(__half)) {
            launchBiasedTanhBackward(nRow,
                                     nCol >> 1,
                                     reinterpret_cast<const __half2*>(grad),
                                     reinterpret_cast<const __half2*>(out),
                                     reinterpret_cast<__half2*>(dInp),
                                     stream);
        }
        else {
            launchBiasedTanhBackward(nRow,
                                     nCol,
                                     grad,
                                     out,
                                     dInp,
                                     stream);
        }

        launchAddBiasOB(nRow,
                        nCol,
                        dInp,
                        dBias,
                        stream);
    }
};

