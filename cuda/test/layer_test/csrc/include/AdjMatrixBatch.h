#pragma once
#include <cuda_runtime.h>


template <typename T>
void
launchAdjMatrixBatch(cudaStream_t stream,
                     const T* inp,
                     T* out,
                     int batch_size,
                     int seq_len,
                     float alpha);

