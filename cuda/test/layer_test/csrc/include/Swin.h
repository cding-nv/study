#pragma once
#include "errMsg.h"


template <typename T, typename I>
__global__ void
windowPartition(int size123,
                int size3,
                const T* __restrict__ inp,
                T* __restrict__ out) {
    const I offset0 = blockIdx.z * I(size123);
    const int offsetS = blockIdx.y * gridDim.x + blockIdx.x;
    const int offsetD = blockIdx.x * gridDim.y + blockIdx.y;

    inp += offsetS * size3 + offset0;
    out += offsetD * size3 + offset0;
    for (int i = threadIdx.x; i < size3; i += blockDim.x) {
        out[i] = __ldg(&inp[i]);
    }
}


template <typename T>
void
launchWindowPartition(int sizeBH,
                      int sizeW,
                      int sizeC,
                      int window_size,
                      const T* inp,
                      T* out,
                      cudaStream_t stream = 0) {
    const int size3 = window_size * sizeC;
    const int size123 = sizeW * window_size * sizeC;
    const int dimBlock = min(1024, size3);
    const dim3 dimGrid(sizeW/window_size, window_size, sizeBH/window_size);
    if (int64_t(sizeBH) * sizeW * sizeC > 2147483647LL) {
        windowPartition<T, int64_t><<<dimGrid, dimBlock, 0, stream>>>(size123, size3, inp, out);
    }
    else {
        windowPartition<T, int><<<dimGrid, dimBlock, 0, stream>>>(size123, size3, inp, out);
    }
}


template <typename T, typename I>
__global__ void
windowReverse(int size123,
              int size3,
              const T* __restrict__ inp,
              T* __restrict__ out) {
    const I offset0 = blockIdx.z * I(size123);
    const int offsetS = blockIdx.x * gridDim.y + blockIdx.y;
    const int offsetD = blockIdx.y * gridDim.x + blockIdx.x;

    inp += offsetS * size3 + offset0;
    out += offsetD * size3 + offset0;
    for (int i = threadIdx.x; i < size3; i += blockDim.x) {
        out[i] = __ldg(&inp[i]);
    }
}


template <typename T>
void
launchWindowReverse(int sizeBH,
                    int sizeW,
                    int sizeC,
                    int window_size,
                    const T* inp,
                    T* out,
                    cudaStream_t stream = 0) {
    const int size3 = window_size * sizeC;
    const int size123 = sizeW * window_size * sizeC;
    const int dimBlock = min(1024, size3);
    const dim3 dimGrid(sizeW/window_size, window_size, sizeBH/window_size);
    if (int64_t(sizeBH) * sizeW * sizeC > 2147483647LL) {
        windowReverse<T, int64_t><<<dimGrid, dimBlock, 0, stream>>>(size123, size3, inp, out);
    }
    else {
        windowReverse<T, int><<<dimGrid, dimBlock, 0, stream>>>(size123, size3, inp, out);
    }
}


template <typename T>
class WindowPartition {
public:
    static void
    forward(const T* inp,
            T* out,
            int sizeB,
            int sizeH,
            int sizeW,
            int sizeC,
            int window_size,
            cudaStream_t stream = 0) {
        if (sizeC & 0x3) {
            errMsg(format("Unsupported EMBED_DIM (%d) for WindowPartition forward", sizeC));
        }
        if (sizeof(T) == 2) {
            launchWindowPartition(sizeB * sizeH,
                                  sizeW,
                                  sizeC >> 2,
                                  window_size,
                                  reinterpret_cast<const float2*>(inp),
                                  reinterpret_cast<float2*>(out),
                                  stream);
        }
        else if (sizeof(T) == 4) {
            launchWindowPartition(sizeB * sizeH,
                                  sizeW,
                                  sizeC >> 2,
                                  window_size,
                                  reinterpret_cast<const float4*>(inp),
                                  reinterpret_cast<float4*>(out),
                                  stream);
        }
        else {
            errMsg("Unsupported datatype for WindowPartition forward");
        }
    }

    static void
    backward(const T* inp,
             T* out,
             int sizeB,
             int sizeH,
             int sizeW,
             int sizeC,
             int window_size,
             cudaStream_t stream = 0) {
        if (sizeC & 0x3) {
            errMsg(format("Unsupported EMBED_DIM (%d) for WindowPartition backward", sizeC));
        }
        if (sizeof(T) == 2) {
            launchWindowReverse(sizeB * sizeH,
                                sizeW,
                                sizeC >> 2,
                                window_size,
                                reinterpret_cast<const float2*>(inp),
                                reinterpret_cast<float2*>(out),
                                stream);
        }
        else if (sizeof(T) == 4) {
            launchWindowReverse(sizeB * sizeH,
                                sizeW,
                                sizeC >> 2,
                                window_size,
                                reinterpret_cast<const float4*>(inp),
                                reinterpret_cast<float4*>(out),
                                stream);
        }
        else {
            errMsg("Unsupported datatype for WindowPartition backward");
        }
    }
};


template <typename T>
class WindowReverse {
public:
    static void
    forward(const T* inp,
            T* out,
            int sizeB,
            int sizeH,
            int sizeW,
            int sizeC,
            int window_size,
            cudaStream_t stream = 0) {
        if (sizeC & 0x3) {
            errMsg(format("Unsupported EMBED_DIM (%d) for WindowReverse forward", sizeC));
        }
        if (sizeof(T) == 2) {
            launchWindowReverse(sizeB * sizeH,
                                sizeW,
                                sizeC >> 2,
                                window_size,
                                reinterpret_cast<const float2*>(inp),
                                reinterpret_cast<float2*>(out),
                                stream);
        }
        else if (sizeof(T) == 4) {
            launchWindowReverse(sizeB * sizeH,
                                sizeW,
                                sizeC >> 2,
                                window_size,
                                reinterpret_cast<const float4*>(inp),
                                reinterpret_cast<float4*>(out),
                                stream);
        }
        else {
            errMsg("Unsupported datatype for WindowReverse forward");
        }
    }

    static void
    backward(const T* inp,
             T* out,
             int sizeB,
             int sizeH,
             int sizeW,
             int sizeC,
             int window_size,
             cudaStream_t stream = 0) {
        if (sizeC & 0x3) {
            errMsg(format("Unsupported EMBED_DIM (%d) for WindowReverse backward", sizeC));
        }
        if (sizeof(T) == 2) {
            launchWindowPartition(sizeB * sizeH,
                                  sizeW,
                                  sizeC >> 2,
                                  window_size,
                                  reinterpret_cast<const float2*>(inp),
                                  reinterpret_cast<float2*>(out),
                                  stream);
        }
        else if (sizeof(T) == 4) {
            launchWindowPartition(sizeB * sizeH,
                                  sizeW,
                                  sizeC >> 2,
                                  window_size,
                                  reinterpret_cast<const float4*>(inp),
                                  reinterpret_cast<float4*>(out),
                                  stream);
        }
        else {
            errMsg("Unsupported datatype for WindowReverse backward");
        }
    }
};


class RelativePositionIndex {
public:
    static void
    run(int height,
        int width,
        int64_t* index,
        cudaStream_t stream = 0);
};

