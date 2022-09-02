#include "SliceSqueeze.h"


__global__ void
sliceSqueezeBackward(int cls_count,
                     int hidden_size,
                     const float* __restrict__ grad,
                     float* __restrict__ dInp) {
    grad += (blockIdx.y * cls_count + blockIdx.x) * hidden_size;
    dInp += (blockIdx.y * gridDim.x + blockIdx.x) * hidden_size;

    if (blockIdx.x < cls_count) {
        for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
            dInp[i] = __ldg(&grad[i]);
        }
    }
    else {
        for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
            dInp[i] = 0.f;
        }
    }
}


__global__ void
sliceSqueezeBackward(int cls_count,
                     int hidden_size,
                     const __half* __restrict__ grad,
                     __half* __restrict__ dInp) {
    const __half2* grad2 = reinterpret_cast<const __half2*>(grad) + (blockIdx.y * cls_count + blockIdx.x) * hidden_size;
    __half2* dInp2 = reinterpret_cast<__half2*>(dInp) + (blockIdx.y * gridDim.x + blockIdx.x) * hidden_size;

    if (blockIdx.x < cls_count) {
        for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
            dInp2[i] = __ldg(&grad2[i]);
        }
    }
    else {
        for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
            dInp2[i] = __float2half2_rn(0.f);
        }
    }
}


// out-of-place sliceSqueeze backward
template <typename T>
void
launchSliceSqueezeBackward(int batch_size,
                           int seq_len,
                           int cls_count,
                           int hidden_size,
                           const T* grad,
                           T* dInp,
                           cudaStream_t stream) {
    if (typeid(T) == typeid(__half)) {
        hidden_size >>= 1;
    }
    const int dimBlock = min(1024, hidden_size >> 2);
    sliceSqueezeBackward<<<dim3(seq_len, batch_size), dimBlock, 0, stream>>>(cls_count, hidden_size, grad, dInp);
}


template
void
launchSliceSqueezeBackward(int batch_size,
                           int seq_len,
                           int cls_count,
                           int hidden_size,
                           const float* grad,
                           float* dInp,
                           cudaStream_t stream);


template
void
launchSliceSqueezeBackward(int batch_size,
                           int seq_len,
                           int cls_count,
                           int hidden_size,
                           const __half* grad,
                           __half* dInp,
                           cudaStream_t stream);

