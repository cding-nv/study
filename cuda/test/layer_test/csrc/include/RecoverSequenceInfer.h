#pragma once
#include <cuda_runtime.h>


class RecoverSequenceInfer {
public:
    template <typename T>
    static void
    forward1D(cudaStream_t stream,
              const T* inp,
              const int64_t* length,
              T* out,
              int batch_size,
              int seq_len);

    template <typename T>
    static void
    forward2D(cudaStream_t stream,
              const T* inp,
              const int64_t* length,
              T* out,
              int batch_size,
              int seq_len,
              int hidden_size);
};

