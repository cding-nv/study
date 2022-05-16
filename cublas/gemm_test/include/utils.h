#pragma once
#include <cuda_runtime.h>
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define GRIDMAX    65536
#define MAXF       3.402823466e+38f


template <typename T>
__host__ __device__ __forceinline__ T
square(T val) {
	return val * val;
}


template <typename T>
__host__ __device__ __forceinline__ T
cubic(T val) {
	return val * val * val;
}


inline unsigned int
ceilDiv(unsigned int numer, unsigned int denom) {
    return (numer + denom - 1) / denom;
}


__host__ __device__ __forceinline__ unsigned int
nextPow2(unsigned int x) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16; 
    return ++x;
}


inline bool
isActive(double prob) {
#ifdef _DROP_TEST
    return prob >= 0.;
#else
    return prob > 0.;
#endif
}


inline float
toScale(double prob) {
    return 1. / (1. - prob);
}

