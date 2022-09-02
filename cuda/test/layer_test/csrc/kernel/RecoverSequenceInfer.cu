#include "RecoverSequenceInfer.h"
#include "RecoverSequenceTrain.h"
#include <cuda_fp16.h>


template <typename T>
void
RecoverSequenceInfer::forward1D(
        cudaStream_t stream,
        const T* inp,
        const int64_t* length,
        T* out,
        int batch_size,
        int seq_len) {
    launchRecoverSequence1D(
            batch_size,
            seq_len,
            inp,
            length,
            out,
            stream);
}


template
void
RecoverSequenceInfer::forward1D(
        cudaStream_t stream,
        const __half* inp,
        const int64_t* length,
        __half* out,
        int batch_size,
        int seq_len);


template
void
RecoverSequenceInfer::forward1D(
        cudaStream_t stream,
        const float* inp,
        const int64_t* length,
        float* out,
        int batch_size,
        int seq_len);


template
void
RecoverSequenceInfer::forward1D(
        cudaStream_t stream,
        const int64_t* inp,
        const int64_t* length,
        int64_t* out,
        int batch_size,
        int seq_len);


template <typename T>
void
RecoverSequenceInfer::forward2D(
        cudaStream_t stream,
        const T* inp,
        const int64_t* length,
        T* out,
        int batch_size,
        int seq_len,
        int hidden_size) {
    launchRecoverSequence2D(
            batch_size,
            seq_len,
            hidden_size,
            inp,
            length,
            out,
            stream);
}


template
void
RecoverSequenceInfer::forward2D(
        cudaStream_t stream,
        const __half* inp,
        const int64_t* length,
        __half* out,
        int batch_size,
        int seq_len,
        int hidden_size);


template
void
RecoverSequenceInfer::forward2D(
        cudaStream_t stream,
        const float* inp,
        const int64_t* length,
        float* out,
        int batch_size,
        int seq_len,
        int hidden_size);


template
void
RecoverSequenceInfer::forward2D(
        cudaStream_t stream,
        const int64_t* inp,
        const int64_t* length,
        int64_t* out,
        int batch_size,
        int seq_len,
        int hidden_size);

