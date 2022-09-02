#pragma once
#include <cuda_runtime.h>


template <typename T>
void
launchMaskLength(cudaStream_t stream,
                 const T* inp,
                 int64_t* length,
                 int batch_size,
                 int seq_len,
                 int sub_len);


template <typename T>
void
launchBatchingSequence(cudaStream_t stream,
                       const T* inp,
                       int64_t* length,
                       T* out,
                       int batch_size,
                       int seq_len,
                       int sub_len);

