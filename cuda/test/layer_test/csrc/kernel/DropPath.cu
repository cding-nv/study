#include "DropPath.h"
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <typeinfo>


__global__ void
dropPathResIFS(int nCol,
               float prob,
               float scale,
               uint64_t seed,
               uint64_t offset,
               const float* __restrict__ residual,
               float* __restrict__ io,
               uint8_t* __restrict__ mask) {
    const float4* residual4 = reinterpret_cast<const float4*>(residual) + blockIdx.x * nCol;
    float4* io4 = reinterpret_cast<float4*>(io) + blockIdx.x * nCol;

    __shared__ uint8_t m;
    if (threadIdx.x == 0) {
        curandStatePhilox4_32_10_t state;
        curand_init(seed, blockIdx.x, offset, &state);
        m = curand_uniform(&state) > prob;
        mask[blockIdx.x] = m;
    }
    __syncthreads();
    scale *= m;

    float4 i4, r4, o4;
    for (int i = threadIdx.x; i < nCol; i += blockDim.x) {
        i4 = io4[i];
        r4 = residual4[i];
        o4.x = i4.x * scale + r4.x;
        o4.y = i4.y * scale + r4.y;
        o4.z = i4.z * scale + r4.z;
        o4.w = i4.w * scale + r4.w;
        io4[i] = o4;
    }
}


__global__ void
dropPathResIFS(int nCol,
               float prob,
               float scale,
               uint64_t seed,
               uint64_t offset,
               const __half* __restrict__ residual,
               __half* __restrict__ io,
               uint8_t* __restrict__ mask) {
    const float4* residual4 = reinterpret_cast<const float4*>(residual) + blockIdx.x * nCol;
    float4* io4 = reinterpret_cast<float4*>(io) + blockIdx.x * nCol;

    __shared__ uint8_t m;
    if (threadIdx.x == 0) {
        curandStatePhilox4_32_10_t state;
        curand_init(seed, blockIdx.x, offset, &state);
        m = curand_uniform(&state) > prob;
        mask[blockIdx.x] = m;
    }
    __syncthreads();
    __half2 scale2 = __float2half2_rn(scale * m);

    float4 i4, r4, o4;
    __half2* i2 = reinterpret_cast<__half2*>(&i4);
    __half2* r2 = reinterpret_cast<__half2*>(&r4);
    __half2* o2 = reinterpret_cast<__half2*>(&o4);
    for (int i = threadIdx.x; i < nCol; i += blockDim.x) {
        i4 = io4[i];
        r4 = residual4[i];
        o2[0] = i2[0] * scale2 + r2[0];
        o2[1] = i2[1] * scale2 + r2[1];
        o2[2] = i2[2] * scale2 + r2[2];
        o2[3] = i2[3] * scale2 + r2[3];
        io4[i] = o4;
    }
}


__global__ void
dropPathResIFL(int hidden_size,
               float prob,
               float scale,
               uint64_t seed,
               uint64_t offset,
               const float* __restrict__ residual,
               float* __restrict__ io,
               uint8_t* __restrict__ mask) {
    int os = (blockIdx.y * gridDim.x + blockIdx.x) * hidden_size;
    const float4* residual4 = reinterpret_cast<const float4*>(residual) + os;
    float4* io4 = reinterpret_cast<float4*>(io) + os;

    __shared__ uint8_t m;
    if (threadIdx.x == 0) {
        curandStatePhilox4_32_10_t state;
        curand_init(seed, blockIdx.y, offset, &state);
        m = curand_uniform(&state) > prob;
        if (blockIdx.x == 0) mask[blockIdx.y] = m;
    }
    __syncthreads();
    scale *= m;

    float4 i4, r4, o4;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        i4 = io4[i];
        r4 = residual4[i];
        o4.x = i4.x * scale + r4.x;
        o4.y = i4.y * scale + r4.y;
        o4.z = i4.z * scale + r4.z;
        o4.w = i4.w * scale + r4.w;
        io4[i] = o4;
    }
}


__global__ void
dropPathResIFL(int hidden_size,
               float prob,
               float scale,
               uint64_t seed,
               uint64_t offset,
               const __half* __restrict__ residual,
               __half* __restrict__ io,
               uint8_t* __restrict__ mask) {
    int os = (blockIdx.y * gridDim.x + blockIdx.x) * hidden_size;
    const float4* residual4 = reinterpret_cast<const float4*>(residual) + os;
    float4* io4 = reinterpret_cast<float4*>(io) + os;

    __shared__ uint8_t m;
    if (threadIdx.x == 0) {
        curandStatePhilox4_32_10_t state;
        curand_init(seed, blockIdx.y, offset, &state);
        m = curand_uniform(&state) > prob;
        if (blockIdx.x == 0) mask[blockIdx.y] = m;
    }
    __syncthreads();
    __half2 scale2 = __float2half2_rn(scale * m);

    float4 i4, r4, o4;
    __half2* i2 = reinterpret_cast<__half2*>(&i4);
    __half2* r2 = reinterpret_cast<__half2*>(&r4);
    __half2* o2 = reinterpret_cast<__half2*>(&o4);
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        i4 = io4[i];
        r4 = residual4[i];
        o2[0] = i2[0] * scale2 + r2[0];
        o2[1] = i2[1] * scale2 + r2[1];
        o2[2] = i2[2] * scale2 + r2[2];
        o2[3] = i2[3] * scale2 + r2[3];
        io4[i] = o4;
    }
}


// drop_path(inp) + res
template <typename T>
void
launchDropPathResIF(int batch_size,
                    int seq_len,
                    int hidden_size,
                    float prob,
                    float scale,
                    uint64_t seed,
                    uint64_t& offset,
                    const T* residual,
                    T* io,
                    uint8_t* mask,
                    cudaStream_t stream) {
    int s = 2 + (typeid(T) == typeid(__half));
    hidden_size >>= s;
    if (hidden_size <= 32) {
        int nCol = seq_len * hidden_size;
        const int dimBlock = min(1024, nCol);
        dropPathResIFS<<<batch_size, dimBlock, 0, stream>>>(nCol, prob, scale, seed, offset, residual, io, mask);
    }
    else {
        const int dimBlock = min(1024, hidden_size);
        const dim3 dimGrid(seq_len, batch_size);
        dropPathResIFL<<<dimGrid, dimBlock, 0, stream>>>(hidden_size, prob, scale, seed, offset, residual, io, mask);
    }
    ++offset;
}


template
void
launchDropPathResIF(int batch_size,
                    int seq_len,
                    int hidden_size,
                    float prob,
                    float scale,
                    uint64_t seed,
                    uint64_t& offset,
                    const float* residual,
                    float* io,
                    uint8_t* mask,
                    cudaStream_t stream);


template
void
launchDropPathResIF(int batch_size,
                    int seq_len,
                    int hidden_size,
                    float prob,
                    float scale,
                    uint64_t seed,
                    uint64_t& offset,
                    const __half* residual,
                    __half* io,
                    uint8_t* mask,
                    cudaStream_t stream);


__global__ void
dropPathOBS(int nCol,
            float scale,
            const float* __restrict__ grad,
            const uint8_t* __restrict__ mask,
            float* __restrict__ dInp) {
    const float4* grad4 = reinterpret_cast<const float4*>(grad) + blockIdx.x * nCol;
    float4* dInp4 = reinterpret_cast<float4*>(dInp) + blockIdx.x * nCol;
    scale *= mask[blockIdx.x];

    float4 i4, o4;
    for (int i = threadIdx.x; i < nCol; i += blockDim.x) {
        i4 = grad4[i];
        o4.x = i4.x * scale;
        o4.y = i4.y * scale;
        o4.z = i4.z * scale;
        o4.w = i4.w * scale;
        dInp4[i] = o4;
    }
}


__global__ void
dropPathOBS(int nCol,
            float scale,
            const __half* __restrict__ grad,
            const uint8_t* __restrict__ mask,
            __half* __restrict__ dInp) {
    const float4* grad4 = reinterpret_cast<const float4*>(grad) + blockIdx.x * nCol;
    float4* dInp4 = reinterpret_cast<float4*>(dInp) + blockIdx.x * nCol;
    __half2 scale2 = __float2half2_rn(scale * mask[blockIdx.x]);

    float4 i4, o4;
    __half2* i2 = reinterpret_cast<__half2*>(&i4);
    __half2* o2 = reinterpret_cast<__half2*>(&o4);
    for (int i = threadIdx.x; i < nCol; i += blockDim.x) {
        i4 = grad4[i];
        o2[0] = i2[0] * scale2;
        o2[1] = i2[1] * scale2;
        o2[2] = i2[2] * scale2;
        o2[3] = i2[3] * scale2;
        dInp4[i] = o4;
    }
}


__global__ void
dropPathOBL(int hidden_size,
            float scale,
            const float* __restrict__ grad,
            const uint8_t* __restrict__ mask,
            float* __restrict__ dInp) {
    int os = (blockIdx.y * gridDim.x + blockIdx.x) * hidden_size;
    const float4* grad4 = reinterpret_cast<const float4*>(grad) + os;
    float4* dInp4 = reinterpret_cast<float4*>(dInp) + os;
    scale *= mask[blockIdx.y];

    float4 i4, o4;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        i4 = grad4[i];
        o4.x = i4.x * scale;
        o4.y = i4.y * scale;
        o4.z = i4.z * scale;
        o4.w = i4.w * scale;
        dInp4[i] = o4;
    }
}


__global__ void
dropPathOBL(int hidden_size,
            float scale,
            const __half* __restrict__ grad,
            const uint8_t* __restrict__ mask,
            __half* __restrict__ dInp) {
    int os = (blockIdx.y * gridDim.x + blockIdx.x) * hidden_size;
    const float4* grad4 = reinterpret_cast<const float4*>(grad) + os;
    float4* dInp4 = reinterpret_cast<float4*>(dInp) + os;
    __half2 scale2 = __float2half2_rn(scale * mask[blockIdx.y]);

    float4 i4, o4;
    __half2* i2 = reinterpret_cast<__half2*>(&i4);
    __half2* o2 = reinterpret_cast<__half2*>(&o4);
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        i4 = grad4[i];
        o2[0] = i2[0] * scale2;
        o2[1] = i2[1] * scale2;
        o2[2] = i2[2] * scale2;
        o2[3] = i2[3] * scale2;
        dInp4[i] = o4;
    }
}


template <typename T>
void
launchDropPathOB(int batch_size,
                 int seq_len,
                 int hidden_size,
                 float scale,
                 const T* grad,
                 const uint8_t* mask,
                 T* dInp,
                 cudaStream_t stream) {
    int s = 2 + (typeid(T) == typeid(__half));
    hidden_size >>= s;
    if (hidden_size <= 32) {
        int nCol = seq_len * hidden_size;
        const int dimBlock = min(1024, nCol);
        dropPathOBS<<<batch_size, dimBlock, 0, stream>>>(nCol, scale, grad, mask, dInp);
    }
    else {
        const int dimBlock = min(1024, hidden_size);
        const dim3 dimGrid(seq_len, batch_size);
        dropPathOBL<<<dimGrid, dimBlock, 0, stream>>>(hidden_size, scale, grad, mask, dInp);
    }
}


template
void
launchDropPathOB(int batch_size,
                 int seq_len,
                 int hidden_size,
                 float scale,
                 const float* grad,
                 const uint8_t* mask,
                 float* dInp,
                 cudaStream_t stream);


template
void
launchDropPathOB(int batch_size,
                 int seq_len,
                 int hidden_size,
                 float scale,
                 const __half* grad,
                 const uint8_t* mask,
                 __half* dInp,
                 cudaStream_t stream);


__global__ void
dropPathBiasResOFS(int seq_len,
                   float prob,
                   float scale,
                   uint64_t seed,
                   uint64_t offset,
                   const float* __restrict__ inp,
                   const float* __restrict__ bias,
                   const float* __restrict__ residual,
                   float* __restrict__ out,
                   uint8_t* __restrict__ mask) {
    int os = blockIdx.x * seq_len * blockDim.x;
    const float4* inp4 = reinterpret_cast<const float4*>(inp) + os;
    const float4* bias4 = reinterpret_cast<const float4*>(bias);
    const float4* residual4 = reinterpret_cast<const float4*>(residual) + os;
    float4* out4 = reinterpret_cast<float4*>(out) + os;

    __shared__ uint8_t m;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        curandStatePhilox4_32_10_t state;
        curand_init(seed, blockIdx.x, offset, &state);
        m = curand_uniform(&state) > prob;
        mask[blockIdx.x] = m;
    }
    __syncthreads();
    scale *= m;

    float4 i4, r4, o4;
    const float4 b4 = bias4[threadIdx.x];
    for (int i = threadIdx.y; i < seq_len; i += blockDim.y) {
        int idx = i * blockDim.x + threadIdx.x;
        i4 = inp4[idx];
        r4 = residual4[idx];
        o4.x = (i4.x + b4.x) * scale + r4.x;
        o4.y = (i4.y + b4.y) * scale + r4.y;
        o4.z = (i4.z + b4.z) * scale + r4.z;
        o4.w = (i4.w + b4.w) * scale + r4.w;
        out4[idx] = o4;
    }
}


__global__ void
dropPathBiasResOFS(int seq_len,
                   float prob,
                   float scale,
                   uint64_t seed,
                   uint64_t offset,
                   const __half* __restrict__ inp,
                   const __half* __restrict__ bias,
                   const __half* __restrict__ residual,
                   __half* __restrict__ out,
                   uint8_t* __restrict__ mask) {
    int os = blockIdx.x * seq_len * blockDim.x;
    const float4* inp4 = reinterpret_cast<const float4*>(inp) + os;
    const float4* bias4 = reinterpret_cast<const float4*>(bias);
    const float4* residual4 = reinterpret_cast<const float4*>(residual) + os;
    float4* out4 = reinterpret_cast<float4*>(out) + os;

    __shared__ uint8_t m;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        curandStatePhilox4_32_10_t state;
        curand_init(seed, blockIdx.x, offset, &state);
        m = curand_uniform(&state) > prob;
        mask[blockIdx.x] = m;
    }
    __syncthreads();
    __half2 scale2 = __float2half2_rn(scale * m);

    float4 i4, r4, o4;
    const float4 b4 = bias4[threadIdx.x];
    __half2* i2 = reinterpret_cast<__half2*>(&i4);
    __half2* r2 = reinterpret_cast<__half2*>(&r4);
    __half2* o2 = reinterpret_cast<__half2*>(&o4);
    const __half2* b2 = reinterpret_cast<const __half2*>(&b4);
    for (int i = threadIdx.y; i < seq_len; i += blockDim.y) {
        int idx = i * blockDim.x + threadIdx.x;
        i4 = inp4[idx];
        r4 = residual4[idx];
        o2[0] = (i2[0] + b2[0]) * scale2 + r2[0];
        o2[1] = (i2[1] + b2[1]) * scale2 + r2[1];
        o2[2] = (i2[2] + b2[2]) * scale2 + r2[2];
        o2[3] = (i2[3] + b2[3]) * scale2 + r2[3];
        out4[idx] = o4;
    }
}


__global__ void
dropPathBiasResOFL(int hidden_size,
                   float prob,
                   float scale,
                   uint64_t seed,
                   uint64_t offset,
                   const float* __restrict__ inp,
                   const float* __restrict__ bias,
                   const float* __restrict__ residual,
                   float* __restrict__ out,
                   uint8_t* __restrict__ mask) {
    int os = (blockIdx.y * gridDim.x + blockIdx.x) * hidden_size;
    const float4* inp4 = reinterpret_cast<const float4*>(inp) + os;
    const float4* bias4 = reinterpret_cast<const float4*>(bias);
    const float4* residual4 = reinterpret_cast<const float4*>(residual) + os;
    float4* out4 = reinterpret_cast<float4*>(out) + os;

    __shared__ uint8_t m;
    if (threadIdx.x == 0) {
        curandStatePhilox4_32_10_t state;
        curand_init(seed, blockIdx.y, offset, &state);
        m = curand_uniform(&state) > prob;
        if (blockIdx.x == 0) mask[blockIdx.y] = m;
    }
    __syncthreads();
    scale *= m;

    float4 i4, b4, r4, o4;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        i4 = inp4[i];
        b4 = bias4[i];
        r4 = residual4[i];
        o4.x = (i4.x + b4.x) * scale + r4.x;
        o4.y = (i4.y + b4.y) * scale + r4.y;
        o4.z = (i4.z + b4.z) * scale + r4.z;
        o4.w = (i4.w + b4.w) * scale + r4.w;
        out4[i] = o4;
    }
}


__global__ void
dropPathBiasResOFL(int hidden_size,
                   float prob,
                   float scale,
                   uint64_t seed,
                   uint64_t offset,
                   const __half* __restrict__ inp,
                   const __half* __restrict__ bias,
                   const __half* __restrict__ residual,
                   __half* __restrict__ out,
                   uint8_t* __restrict__ mask) {
    int os = (blockIdx.y * gridDim.x + blockIdx.x) * hidden_size;
    const float4* inp4 = reinterpret_cast<const float4*>(inp) + os;
    const float4* bias4 = reinterpret_cast<const float4*>(bias);
    const float4* residual4 = reinterpret_cast<const float4*>(residual) + os;
    float4* out4 = reinterpret_cast<float4*>(out) + os;

    __shared__ uint8_t m;
    if (threadIdx.x == 0) {
        curandStatePhilox4_32_10_t state;
        curand_init(seed, blockIdx.y, offset, &state);
        m = curand_uniform(&state) > prob;
        if (blockIdx.x == 0) mask[blockIdx.y] = m;
    }
    __syncthreads();
    __half2 scale2 = __float2half2_rn(scale * m);

    float4 i4, b4, r4, o4;
    __half2* i2 = reinterpret_cast<__half2*>(&i4);
    __half2* b2 = reinterpret_cast<__half2*>(&b4);
    __half2* r2 = reinterpret_cast<__half2*>(&r4);
    __half2* o2 = reinterpret_cast<__half2*>(&o4);
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        i4 = inp4[i];
        b4 = bias4[i];
        r4 = residual4[i];
        o2[0] = (i2[0] + b2[0]) * scale2 + r2[0];
        o2[1] = (i2[1] + b2[1]) * scale2 + r2[1];
        o2[2] = (i2[2] + b2[2]) * scale2 + r2[2];
        o2[3] = (i2[3] + b2[3]) * scale2 + r2[3];
        out4[i] = o4;
    }
}


// drop_path(inp + bias) + res
template <typename T>
void
launchDropPathBiasResOF(int batch_size,
                        int seq_len,
                        int hidden_size,
                        float prob,
                        float scale,
                        uint64_t seed,
                        uint64_t& offset,
                        const T* inp,
                        const T* bias,
                        const T* residual,
                        T* out,
                        uint8_t* mask,
                        cudaStream_t stream) {
    int s = 2 + (typeid(T) == typeid(__half));
    hidden_size >>= s;
    if (hidden_size <= 32) {
        const dim3 dimBlock(hidden_size, min(seq_len, 1024 / hidden_size));
        dropPathBiasResOFS<<<batch_size, dimBlock, 0, stream>>>(seq_len, prob, scale, seed, offset, inp, bias, residual, out, mask);
    }
    else {
        const int dimBlock = min(1024, hidden_size);
        const dim3 dimGrid(seq_len, batch_size);
        dropPathBiasResOFL<<<dimGrid, dimBlock, 0, stream>>>(hidden_size, prob, scale, seed, offset, inp, bias, residual, out, mask);
    }
    ++offset;
}


template
void
launchDropPathBiasResOF(int batch_size,
                        int seq_len,
                        int hidden_size,
                        float prob,
                        float scale,
                        uint64_t seed,
                        uint64_t& offset,
                        const float* inp,
                        const float* bias,
                        const float* residual,
                        float* out,
                        uint8_t* mask,
                        cudaStream_t stream);


template
void
launchDropPathBiasResOF(int batch_size,
                        int seq_len,
                        int hidden_size,
                        float prob,
                        float scale,
                        uint64_t seed,
                        uint64_t& offset,
                        const __half* inp,
                        const __half* bias,
                        const __half* residual,
                        __half* out,
                        uint8_t* mask,
                        cudaStream_t stream);

