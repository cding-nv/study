#include "Dropout.h"
#include "utils.h"
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <typeinfo>


__global__ void
dropoutIF(float prob,
          float scale,
          uint64_t seed,
          uint64_t offset,
          int N,
          float* __restrict__ io,
          uint8_t* __restrict__ mask) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 
    if (tid >= N) return;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, tid, offset, &state);
    tid <<= 2;

    uint8_t m[4];
    float4 rand4 = curand_uniform4(&state);
    m[0] = rand4.x > prob;
    m[1] = rand4.y > prob;
    m[2] = rand4.z > prob;
    m[3] = rand4.w > prob;
    *reinterpret_cast<uint32_t*>(&mask[tid]) = *reinterpret_cast<uint32_t*>(m);

    float4 io4 = *reinterpret_cast<float4*>(&io[tid]);
    io4.x *= scale * m[0];
    io4.y *= scale * m[1];
    io4.z *= scale * m[2];
    io4.w *= scale * m[3];
    *reinterpret_cast<float4*>(&io[tid]) = io4;
}


__global__ void
dropoutIF(float prob,
          float scale,
          uint64_t seed,
          uint64_t offset,
          int N,
          __half* __restrict__ io,
          uint8_t* __restrict__ mask) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 
    if (tid >= N) return;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, tid, offset, &state);
    const __half2 scale2 = __float2half2_rn(scale);
    tid <<= 3;

    uint8_t m[8];
    float4 rand4 = curand_uniform4(&state);
    m[0] = rand4.x > prob;
    m[1] = rand4.y > prob;
    m[2] = rand4.z > prob;
    m[3] = rand4.w > prob;
    rand4 = curand_uniform4(&state);
    m[4] = rand4.x > prob;
    m[5] = rand4.y > prob;
    m[6] = rand4.z > prob;
    m[7] = rand4.w > prob;
    *reinterpret_cast<uint64_t*>(&mask[tid]) = *reinterpret_cast<uint64_t*>(m);

    float4 io4 = *reinterpret_cast<float4*>(&io[tid]);
    __half2* io2 = reinterpret_cast<__half2*>(&io4);
    __half2 m20(m[0], m[1]);
    __half2 m21(m[2], m[3]);
    __half2 m22(m[4], m[5]);
    __half2 m23(m[6], m[7]);
    io2[0] *= scale2 * m20;
    io2[1] *= scale2 * m21;
    io2[2] *= scale2 * m22;
    io2[3] *= scale2 * m23;
    *reinterpret_cast<float4*>(&io[tid]) = io4;
}


template <typename T>
void
launchDropoutIF(float prob,
                float scale,
                uint64_t seed,
                uint64_t& offset,
                int N,
                T* io,
                uint8_t* mask,
                cudaStream_t stream) {
    int s = 2 + (typeid(T) == typeid(__half));
    N >>= s;
    constexpr int dimBlock = 1024;
    const int dimGrid = ceilDiv(N, dimBlock);
    dropoutIF<<<dimGrid, dimBlock, 0, stream>>>(prob, scale, seed, offset, N, io, mask);
    offset += 1 << s;
}


template
void
launchDropoutIF(float prob,
                float scale,
                uint64_t seed,
                uint64_t& offset,
                int N,
                float* io,
                uint8_t* mask,
                cudaStream_t stream);


template
void
launchDropoutIF(float prob,
                float scale,
                uint64_t seed,
                uint64_t& offset,
                int N,
                __half* io,
                uint8_t* mask,
                cudaStream_t stream);


__global__ void
dropoutOF(float prob,
          float scale,
          uint64_t seed,
          uint64_t offset,
          int N,
          const float* __restrict__ inp,
          float* __restrict__ out,
          uint8_t* __restrict__ mask) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 
    if (tid >= N) return;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, tid, offset, &state);
    tid <<= 2;

    uint8_t m[4];
    float4 rand4 = curand_uniform4(&state);
    m[0] = rand4.x > prob;
    m[1] = rand4.y > prob;
    m[2] = rand4.z > prob;
    m[3] = rand4.w > prob;
    *reinterpret_cast<uint32_t*>(&mask[tid]) = *reinterpret_cast<uint32_t*>(m);

    float4 o4;
    const float4 i4 = *reinterpret_cast<const float4*>(&inp[tid]);
    o4.x = i4.x * scale * m[0];
    o4.y = i4.y * scale * m[1];
    o4.z = i4.z * scale * m[2];
    o4.w = i4.w * scale * m[3];
    *reinterpret_cast<float4*>(&out[tid]) = o4;
}


__global__ void
dropoutOF(float prob,
          float scale,
          uint64_t seed,
          uint64_t offset,
          int N,
          const __half* __restrict__ inp,
          __half* __restrict__ out,
          uint8_t* __restrict__ mask) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 
    if (tid >= N) return;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, tid, offset, &state);
    const __half2 scale2 = __float2half2_rn(scale);
    tid <<= 3;

    uint8_t m[8];
    float4 rand4 = curand_uniform4(&state);
    m[0] = rand4.x > prob;
    m[1] = rand4.y > prob;
    m[2] = rand4.z > prob;
    m[3] = rand4.w > prob;
    rand4 = curand_uniform4(&state);
    m[4] = rand4.x > prob;
    m[5] = rand4.y > prob;
    m[6] = rand4.z > prob;
    m[7] = rand4.w > prob;
    *reinterpret_cast<uint64_t*>(&mask[tid]) = *reinterpret_cast<uint64_t*>(m);

    __half2 o2[4];
    const float4 i4 = *reinterpret_cast<const float4*>(&inp[tid]);
    const __half2* i2 = reinterpret_cast<const __half2*>(&i4);
    __half2 m20(m[0], m[1]);
    __half2 m21(m[2], m[3]);
    __half2 m22(m[4], m[5]);
    __half2 m23(m[6], m[7]);
    o2[0] = i2[0] * scale2 * m20;
    o2[1] = i2[1] * scale2 * m21;
    o2[2] = i2[2] * scale2 * m22;
    o2[3] = i2[3] * scale2 * m23;
    *reinterpret_cast<float4*>(&out[tid]) = *reinterpret_cast<float4*>(o2);
}


template <typename T>
void
launchDropoutOF(float prob,
                float scale,
                uint64_t seed,
                uint64_t& offset,
                int N,
                const T* inp,
                T* out,
                uint8_t* mask,
                cudaStream_t stream) {
    int s = 2 + (typeid(T) == typeid(__half));
    N >>= s;
    constexpr int dimBlock = 1024;
    const int dimGrid = ceilDiv(N, dimBlock);
    dropoutOF<<<dimGrid, dimBlock, 0, stream>>>(prob, scale, seed, offset, N, inp, out, mask);
    offset += 1 << s;
}


template
void
launchDropoutOF(float prob,
                float scale,
                uint64_t seed,
                uint64_t& offset,
                int N,
                const float* inp,
                float* out,
                uint8_t* mask,
                cudaStream_t stream);


template
void
launchDropoutOF(float prob,
                float scale,
                uint64_t seed,
                uint64_t& offset,
                int N,
                const __half* inp,
                __half* out,
                uint8_t* mask,
                cudaStream_t stream);


__global__ void
dropoutIB(float scale,
          int N,
          float* __restrict__ grad,
          const uint8_t* __restrict__ mask) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 
    if (tid >= N) return;
    tid <<= 2;

    const uint32_t m4 = *reinterpret_cast<const uint32_t*>(&mask[tid]);
    float4 grad4 = *reinterpret_cast<float4*>(&grad[tid]);
    const uint8_t* m = reinterpret_cast<const uint8_t*>(&m4);
    grad4.x *= scale * m[0];
    grad4.y *= scale * m[1];
    grad4.z *= scale * m[2];
    grad4.w *= scale * m[3];
    *reinterpret_cast<float4*>(&grad[tid]) = grad4;
}


__global__ void
dropoutIB(float scale,
          int N,
          __half* __restrict__ grad,
          const uint8_t* __restrict__ mask) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 
    if (tid >= N) return;
    const __half2 scale2 = __float2half2_rn(scale);
    tid <<= 3;

    const uint64_t m8 = *reinterpret_cast<const uint64_t*>(&mask[tid]);
    float4 grad4 = *reinterpret_cast<float4*>(&grad[tid]);
    const uint8_t* m = reinterpret_cast<const uint8_t*>(&m8);
    __half2* g2 = reinterpret_cast<__half2*>(&grad4);
    __half2 m20(m[0], m[1]);
    __half2 m21(m[2], m[3]);
    __half2 m22(m[4], m[5]);
    __half2 m23(m[6], m[7]);
    g2[0] *= scale2 * m20;
    g2[1] *= scale2 * m21;
    g2[2] *= scale2 * m22;
    g2[3] *= scale2 * m23;
    *reinterpret_cast<float4*>(&grad[tid]) = grad4;
}


template <typename T>
void
launchDropoutIB(float scale,
                int N,
                T* grad,
                const uint8_t* mask,
                cudaStream_t stream) {
    int s = 2 + (typeid(T) == typeid(__half));
    N >>= s;
    constexpr int dimBlock = 1024;
    const int dimGrid = ceilDiv(N, dimBlock);
    dropoutIB<<<dimGrid, dimBlock, 0, stream>>>(scale, N, grad, mask);
}


template
void
launchDropoutIB(float scale,
                int N,
                float* grad,
                const uint8_t* mask,
                cudaStream_t stream);


template
void
launchDropoutIB(float scale,
                int N,
                __half* grad,
                const uint8_t* mask,
                cudaStream_t stream);


__global__ void
dropoutOB(float scale,
          int N,
          const float* __restrict__ grad,
          const uint8_t* __restrict__ mask,
          float* __restrict dInp) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 
    if (tid >= N) return;
    tid <<= 2;

    float4 o4;
    const uint32_t m4 = *reinterpret_cast<const uint32_t*>(&mask[tid]);
    const float4 grad4 = *reinterpret_cast<const float4*>(&grad[tid]);
    const uint8_t* m = reinterpret_cast<const uint8_t*>(&m4);
    o4.x = grad4.x * scale * m[0];
    o4.y = grad4.y * scale * m[1];
    o4.z = grad4.z * scale * m[2];
    o4.w = grad4.w * scale * m[3];
    *reinterpret_cast<float4*>(&dInp[tid]) = o4;
}


__global__ void
dropoutOB(float scale,
          int N,
          const __half* __restrict__ grad,
          const uint8_t* __restrict__ mask,
          __half* __restrict dInp) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 
    if (tid >= N) return;
    const __half2 scale2 = __float2half2_rn(scale);
    tid <<= 3;

    __half2 o2[4];
    const uint64_t m8 = *reinterpret_cast<const uint64_t*>(&mask[tid]);
    const float4 grad4 = *reinterpret_cast<const float4*>(&grad[tid]);
    const uint8_t* m = reinterpret_cast<const uint8_t*>(&m8);
    const __half2* g2 = reinterpret_cast<const __half2*>(&grad4);
    __half2 m20(m[0], m[1]);
    __half2 m21(m[2], m[3]);
    __half2 m22(m[4], m[5]);
    __half2 m23(m[6], m[7]);
    o2[0] = g2[0] * scale2 * m20;
    o2[1] = g2[1] * scale2 * m21;
    o2[2] = g2[2] * scale2 * m22;
    o2[3] = g2[3] * scale2 * m23;
    *reinterpret_cast<float4*>(&dInp[tid]) = *reinterpret_cast<float4*>(o2);
}


template <typename T>
void
launchDropoutOB(float scale,
                int N,
                const T* grad,
                const uint8_t* mask,
                T* dInp,
                cudaStream_t stream) {
    int s = 2 + (typeid(T) == typeid(__half));
    N >>= s;
    constexpr int dimBlock = 1024;
    const int dimGrid = ceilDiv(N, dimBlock);
    dropoutOB<<<dimGrid, dimBlock, 0, stream>>>(scale, N, grad, mask, dInp);
}


template
void
launchDropoutOB(float scale,
                int N,
                const float* grad,
                const uint8_t* mask,
                float* dInp,
                cudaStream_t stream);


template
void
launchDropoutOB(float scale,
                int N,
                const __half* grad,
                const uint8_t* mask,
                __half* dInp,
                cudaStream_t stream);


__global__ void
dropoutBiasIF(float prob,
              float scale,
              uint64_t seed,
              uint64_t offset,
              int nCol,
              const float* __restrict__ bias,
              float* __restrict__ io,
              uint8_t* __restrict__ mask) {
    const float4* bias4 = reinterpret_cast<const float4*>(bias);
    float4* io4 = reinterpret_cast<float4*>(io) + blockIdx.x * nCol;
    uint32_t* mask4 = reinterpret_cast<uint32_t*>(mask) + blockIdx.x * nCol;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, blockIdx.x * blockDim.x + threadIdx.x, offset, &state);

    uint32_t m4;
    float4 rand4, i4, b4, o4;
    uint8_t* m = reinterpret_cast<uint8_t*>(&m4);
    for (int i = threadIdx.x; i < nCol; i += blockDim.x) {
        rand4 = curand_uniform4(&state);
        m[0] = rand4.x > prob;
        m[1] = rand4.y > prob;
        m[2] = rand4.z > prob;
        m[3] = rand4.w > prob;
        mask4[i] = m4;

        i4 = io4[i];
        b4 = bias4[i];
        o4.x = (i4.x + b4.x) * scale * m[0];
        o4.y = (i4.y + b4.y) * scale * m[1];
        o4.z = (i4.z + b4.z) * scale * m[2];
        o4.w = (i4.w + b4.w) * scale * m[3];
        io4[i] = o4;
    }
}


__global__ void
dropoutBiasIF(float prob,
              float scale,
              uint64_t seed,
              uint64_t offset,
              int nCol,
              const __half* __restrict__ bias,
              __half* __restrict__ io,
              uint8_t* __restrict__ mask) {
    const float4* bias4 = reinterpret_cast<const float4*>(bias);
    float4* io4 = reinterpret_cast<float4*>(io) + blockIdx.x * nCol;
    uint64_t* mask8 = reinterpret_cast<uint64_t*>(mask) + blockIdx.x * nCol;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, blockIdx.x * blockDim.x + threadIdx.x, offset, &state);
    const __half2 scale2 = __float2half2_rn(scale);

    uint64_t m8;
    float4 rand4, i4, b4, o4;
    uint8_t* m = reinterpret_cast<uint8_t*>(&m8);
    __half2* i2 = reinterpret_cast<__half2*>(&i4);
    __half2* b2 = reinterpret_cast<__half2*>(&b4);
    __half2* o2 = reinterpret_cast<__half2*>(&o4);
    for (int i = threadIdx.x; i < nCol; i += blockDim.x) {
        rand4 = curand_uniform4(&state);
        m[0] = rand4.x > prob;
        m[1] = rand4.y > prob;
        m[2] = rand4.z > prob;
        m[3] = rand4.w > prob;
        rand4 = curand_uniform4(&state);
        m[4] = rand4.x > prob;
        m[5] = rand4.y > prob;
        m[6] = rand4.z > prob;
        m[7] = rand4.w > prob;
        mask8[i] = m8;

        i4 = io4[i];
        b4 = bias4[i];
        __half2 m20(m[0], m[1]);
        __half2 m21(m[2], m[3]);
        __half2 m22(m[4], m[5]);
        __half2 m23(m[6], m[7]);
        o2[0] = (i2[0] + b2[0]) * scale2 * m20;
        o2[1] = (i2[1] + b2[1]) * scale2 * m21;
        o2[2] = (i2[2] + b2[2]) * scale2 * m22;
        o2[3] = (i2[3] + b2[3]) * scale2 * m23;
        io4[i] = o4;
    }
}


// dropout(inp + bias)
template <typename T>
void
launchDropoutBiasIF(float prob,
                    float scale,
                    uint64_t seed,
                    uint64_t& offset,
                    int nRow,
                    int nCol,
                    const T* bias,
                    T* io,
                    uint8_t* mask,
                    cudaStream_t stream) {
    int s = 2 + (typeid(T) == typeid(__half));
    nCol >>= s;
    const int dimBlock = min(1024, nCol);
    dropoutBiasIF<<<nRow, dimBlock, 0, stream>>>(prob, scale, seed, offset, nCol, bias, io, mask);
    offset += (1 << s) * ((nCol + 1023) >> 10);
}


template
void
launchDropoutBiasIF(float prob,
                    float scale,
                    uint64_t seed,
                    uint64_t& offset,
                    int nRow,
                    int nCol,
                    const float* bias,
                    float* io,
                    uint8_t* mask,
                    cudaStream_t stream);


template
void
launchDropoutBiasIF(float prob,
                    float scale,
                    uint64_t seed,
                    uint64_t& offset,
                    int nRow,
                    int nCol,
                    const __half* bias,
                    __half* io,
                    uint8_t* mask,
                    cudaStream_t stream);


__global__ void
dropoutBiasOF(float prob,
              float scale,
              uint64_t seed,
              uint64_t offset,
              int nCol,
              const float* __restrict__ inp,
              const float* __restrict__ bias,
              float* __restrict__ out,
              uint8_t* __restrict__ mask) {
    const float4* inp4 = reinterpret_cast<const float4*>(inp) + blockIdx.x * nCol;
    const float4* bias4 = reinterpret_cast<const float4*>(bias);
    float4* out4 = reinterpret_cast<float4*>(out) + blockIdx.x * nCol;
    uint32_t* mask4 = reinterpret_cast<uint32_t*>(mask) + blockIdx.x * nCol;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, blockIdx.x * blockDim.x + threadIdx.x, offset, &state);

    uint32_t m4;
    float4 rand4, i4, b4, o4;
    uint8_t* m = reinterpret_cast<uint8_t*>(&m4);
    for (int i = threadIdx.x; i < nCol; i += blockDim.x) {
        rand4 = curand_uniform4(&state);
        m[0] = rand4.x > prob;
        m[1] = rand4.y > prob;
        m[2] = rand4.z > prob;
        m[3] = rand4.w > prob;
        mask4[i] = m4;

        i4 = inp4[i];
        b4 = bias4[i];
        o4.x = (i4.x + b4.x) * scale * m[0];
        o4.y = (i4.y + b4.y) * scale * m[1];
        o4.z = (i4.z + b4.z) * scale * m[2];
        o4.w = (i4.w + b4.w) * scale * m[3];
        out4[i] = o4;
    }
}


__global__ void
dropoutBiasOF(float prob,
              float scale,
              uint64_t seed,
              uint64_t offset,
              int nCol,
              const __half* __restrict__ inp,
              const __half* __restrict__ bias,
              __half* __restrict__ out,
              uint8_t* __restrict__ mask) {
    const float4* inp4 = reinterpret_cast<const float4*>(inp) + blockIdx.x * nCol;
    const float4* bias4 = reinterpret_cast<const float4*>(bias);
    float4* out4 = reinterpret_cast<float4*>(out) + blockIdx.x * nCol;
    uint64_t* mask8 = reinterpret_cast<uint64_t*>(mask) + blockIdx.x * nCol;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, blockIdx.x * blockDim.x + threadIdx.x, offset, &state);
    const __half2 scale2 = __float2half2_rn(scale);

    uint64_t m8;
    float4 rand4, i4, b4, o4;
    uint8_t* m = reinterpret_cast<uint8_t*>(&m8);
    __half2* i2 = reinterpret_cast<__half2*>(&i4);
    __half2* b2 = reinterpret_cast<__half2*>(&b4);
    __half2* o2 = reinterpret_cast<__half2*>(&o4);
    for (int i = threadIdx.x; i < nCol; i += blockDim.x) {
        rand4 = curand_uniform4(&state);
        m[0] = rand4.x > prob;
        m[1] = rand4.y > prob;
        m[2] = rand4.z > prob;
        m[3] = rand4.w > prob;
        rand4 = curand_uniform4(&state);
        m[4] = rand4.x > prob;
        m[5] = rand4.y > prob;
        m[6] = rand4.z > prob;
        m[7] = rand4.w > prob;
        mask8[i] = m8;

        i4 = inp4[i];
        b4 = bias4[i];
        __half2 m20(m[0], m[1]);
        __half2 m21(m[2], m[3]);
        __half2 m22(m[4], m[5]);
        __half2 m23(m[6], m[7]);
        o2[0] = (i2[0] + b2[0]) * scale2 * m20;
        o2[1] = (i2[1] + b2[1]) * scale2 * m21;
        o2[2] = (i2[2] + b2[2]) * scale2 * m22;
        o2[3] = (i2[3] + b2[3]) * scale2 * m23;
        out4[i] = o4;
    }
}


// dropout(inp + bias)
template <typename T>
void
launchDropoutBiasOF(float prob,
                    float scale,
                    uint64_t seed,
                    uint64_t& offset,
                    int nRow,
                    int nCol,
                    const T* inp,
                    const T* bias,
                    T* out,
                    uint8_t* mask,
                    cudaStream_t stream) {
    int s = 2 + (typeid(T) == typeid(__half));
    nCol >>= s;
    const int dimBlock = min(1024, nCol);
    dropoutBiasOF<<<nRow, dimBlock, 0, stream>>>(prob, scale, seed, offset, nCol, inp, bias, out, mask);
    offset += (1 << s) * ((nCol + 1023) >> 10);
}


template
void
launchDropoutBiasOF(float prob,
                    float scale,
                    uint64_t seed,
                    uint64_t& offset,
                    int nRow,
                    int nCol,
                    const float* inp,
                    const float* bias,
                    float* out,
                    uint8_t* mask,
                    cudaStream_t stream);


template
void
launchDropoutBiasOF(float prob,
                    float scale,
                    uint64_t seed,
                    uint64_t& offset,
                    int nRow,
                    int nCol,
                    const __half* inp,
                    const __half* bias,
                    __half* out,
                    uint8_t* mask,
                    cudaStream_t stream);


__global__ void
dropoutBiasResOF(float prob,
                 float scale,
                 uint64_t seed,
                 uint64_t offset,
                 int nCol,
                 const float* __restrict__ inp,
                 const float* __restrict__ bias,
                 const float* __restrict__ residual,
                 float* __restrict__ out,
                 uint8_t* __restrict__ mask) {
    const float4* inp4 = reinterpret_cast<const float4*>(inp) + blockIdx.x * nCol;
    const float4* bias4 = reinterpret_cast<const float4*>(bias);
    const float4* residual4 = reinterpret_cast<const float4*>(residual) + blockIdx.x * nCol;
    float4* out4 = reinterpret_cast<float4*>(out) + blockIdx.x * nCol;
    uint32_t* mask4 = reinterpret_cast<uint32_t*>(mask) + blockIdx.x * nCol;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, blockIdx.x * blockDim.x + threadIdx.x, offset, &state);

    uint32_t m4;
    float4 rand4, i4, b4, r4, o4;
    uint8_t* m = reinterpret_cast<uint8_t*>(&m4);
    for (int i = threadIdx.x; i < nCol; i += blockDim.x) {
        rand4 = curand_uniform4(&state);
        m[0] = rand4.x > prob;
        m[1] = rand4.y > prob;
        m[2] = rand4.z > prob;
        m[3] = rand4.w > prob;
        mask4[i] = m4;

        i4 = inp4[i];
        b4 = bias4[i];
        r4 = residual4[i];
        o4.x = (i4.x + b4.x) * scale * m[0] + r4.x;
        o4.y = (i4.y + b4.y) * scale * m[1] + r4.y;
        o4.z = (i4.z + b4.z) * scale * m[2] + r4.z;
        o4.w = (i4.w + b4.w) * scale * m[3] + r4.w;
        out4[i] = o4;
    }
}


__global__ void
dropoutBiasResOF(float prob,
                 float scale,
                 uint64_t seed,
                 uint64_t offset,
                 int nCol,
                 const __half* __restrict__ inp,
                 const __half* __restrict__ bias,
                 const __half* __restrict__ residual,
                 __half* __restrict__ out,
                 uint8_t* __restrict__ mask) {
    const float4* inp4 = reinterpret_cast<const float4*>(inp) + blockIdx.x * nCol;
    const float4* bias4 = reinterpret_cast<const float4*>(bias);
    const float4* residual4 = reinterpret_cast<const float4*>(residual) + blockIdx.x * nCol;
    float4* out4 = reinterpret_cast<float4*>(out) + blockIdx.x * nCol;
    uint64_t* mask8 = reinterpret_cast<uint64_t*>(mask) + blockIdx.x * nCol;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, blockIdx.x * blockDim.x + threadIdx.x, offset, &state);
    const __half2 scale2 = __float2half2_rn(scale);

    uint64_t m8;
    float4 rand4, i4, b4, r4, o4;
    uint8_t* m = reinterpret_cast<uint8_t*>(&m8);
    __half2* i2 = reinterpret_cast<__half2*>(&i4);
    __half2* b2 = reinterpret_cast<__half2*>(&b4);
    __half2* r2 = reinterpret_cast<__half2*>(&r4);
    __half2* o2 = reinterpret_cast<__half2*>(&o4);
    for (int i = threadIdx.x; i < nCol; i += blockDim.x) {
        rand4 = curand_uniform4(&state);
        m[0] = rand4.x > prob;
        m[1] = rand4.y > prob;
        m[2] = rand4.z > prob;
        m[3] = rand4.w > prob;
        rand4 = curand_uniform4(&state);
        m[4] = rand4.x > prob;
        m[5] = rand4.y > prob;
        m[6] = rand4.z > prob;
        m[7] = rand4.w > prob;
        mask8[i] = m8;

        i4 = inp4[i];
        b4 = bias4[i];
        r4 = residual4[i];
        __half2 m20(m[0], m[1]);
        __half2 m21(m[2], m[3]);
        __half2 m22(m[4], m[5]);
        __half2 m23(m[6], m[7]);
        o2[0] = (i2[0] + b2[0]) * scale2 * m20 + r2[0];
        o2[1] = (i2[1] + b2[1]) * scale2 * m21 + r2[1];
        o2[2] = (i2[2] + b2[2]) * scale2 * m22 + r2[2];
        o2[3] = (i2[3] + b2[3]) * scale2 * m23 + r2[3];
        out4[i] = o4;
    }
}


// dropout(inp + bias) + res
template <typename T>
void
launchDropoutBiasResOF(float prob,
                       float scale,
                       uint64_t seed,
                       uint64_t& offset,
                       int nRow,
                       int nCol,
                       const T* inp,
                       const T* bias,
                       const T* residual,
                       T* out,
                       uint8_t* mask,
                       cudaStream_t stream) {
    int s = 2 + (typeid(T) == typeid(__half));
    nCol >>= s;
    const int dimBlock = min(1024, nCol);
    dropoutBiasResOF<<<nRow, dimBlock, 0, stream>>>(prob, scale, seed, offset, nCol, inp, bias, residual, out, mask);
    offset += (1 << s) * ((nCol + 1023) >> 10);
}


template
void
launchDropoutBiasResOF(float prob,
                       float scale,
                       uint64_t seed,
                       uint64_t& offset,
                       int nRow,
                       int nCol,
                       const float* inp,
                       const float* bias,
                       const float* residual,
                       float* out,
                       uint8_t* mask,
                       cudaStream_t stream);


template
void
launchDropoutBiasResOF(float prob,
                       float scale,
                       uint64_t seed,
                       uint64_t& offset,
                       int nRow,
                       int nCol,
                       const __half* inp,
                       const __half* bias,
                       const __half* residual,
                       __half* out,
                       uint8_t* mask,
                       cudaStream_t stream);

