#pragma once
#include <cuda_fp16.h>
#include <typeinfo>


template <typename T>
__global__ void
sliceSqueeze(int seq_len,
             int hidden_size,
             const T* __restrict__ inp,
             T* __restrict__ out) {
    inp += (blockIdx.y * seq_len + blockIdx.x) * hidden_size;
    out += (blockIdx.y * gridDim.x + blockIdx.x) * hidden_size;

    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        out[i] = __ldg(&inp[i]);
    }
}


// out-of-place sliceSqueeze forward
template <typename T>
inline void
launchSliceSqueeze(int batch_size,
                   int seq_len,
                   int cls_count,
                   int hidden_size,
                   const T* inp,
                   T* out,
                   cudaStream_t stream = 0) {
    const int dimBlock = min(1024, hidden_size >> 2);
    sliceSqueeze<<<dim3(cls_count, batch_size), dimBlock, 0, stream>>>(seq_len, hidden_size, inp, out);
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
                           cudaStream_t stream = 0);


template <typename T>
class SliceSqueeze {
public:
    static void
    forward(const T* inp,
            T* out,
            int batch_size,
            int seq_len,
            int hidden_size,
            int cls_count,
            cudaStream_t stream = 0) {
        if (typeid(T) == typeid(__half) && !(hidden_size & 1)) {
            launchSliceSqueeze(batch_size,
                               seq_len,
                               cls_count,
                               hidden_size >> 1,
                               reinterpret_cast<const __half2*>(inp),
                               reinterpret_cast<__half2*>(out),
                               stream);
        }
        else {
            launchSliceSqueeze(batch_size,
                               seq_len,
                               cls_count,
                               hidden_size,
                               inp,
                               out,
                               stream);
        }
    }

    static void
    backward(const T* grad,
             T* dInp,
             int batch_size,
             int seq_len,
             int hidden_size,
             int cls_count,
             cudaStream_t stream = 0) {
        launchSliceSqueezeBackward(batch_size,
                                   seq_len,
                                   cls_count,
                                   hidden_size,
                                   grad,
                                   dInp,
                                   stream);
    }
};

