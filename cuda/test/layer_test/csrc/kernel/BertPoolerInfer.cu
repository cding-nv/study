#include "BertPoolerInfer.h"
#include "BertCom.h"
#include "BiasedTanh.h"
#include "Context.h"
#include "Linear.h"
#include "SliceSqueeze.h"


int
BertPoolerInfer::get_f_buff_size(
        int batch_size,
        int hidden_size,
        int cls_count) {
    int output_numel = BertCom::get_output_numel(batch_size, hidden_size, cls_count);
    return output_numel;
}


template <typename T>
void
BertPoolerInfer::forward(
        cudaStream_t stream,
        const T* hidden_states,
        const T* linear_weight,
        const T* linear_bias,
        T* pooled_output,
        T* buffer,
        int batch_size,
        int seq_len,
        int hidden_size,
        int cls_count,
        bool fast_fp32) {
    int batch_cnt = batch_size * cls_count;
    cublasHandle_t handle = Context::instance().getHandle(stream, fast_fp32);

    SliceSqueeze<T>::forward(
            hidden_states,
            buffer,                     // output_numel, out
            batch_size,
            seq_len,
            hidden_size,
            cls_count,
            stream);

    Linear<T>::forward(
            buffer,                     // output_numel, in
            linear_weight,
            pooled_output,
            handle,
            batch_cnt,
            hidden_size,
            hidden_size);

    BiasedTanh<T>::forward(
            linear_bias,
            pooled_output,
            batch_cnt,
            hidden_size,
            stream);
}


template
void
BertPoolerInfer::forward(
        cudaStream_t stream,
        const float* hidden_states,
        const float* linear_weight,
        const float* linear_bias,
        float* pooled_output,
        float* buffer,
        int batch_size,
        int seq_len,
        int hidden_size,
        int cls_count,
        bool fast_fp32);


template
void
BertPoolerInfer::forward(
        cudaStream_t stream,
        const __half* hidden_states,
        const __half* linear_weight,
        const __half* linear_bias,
        __half* pooled_output,
        __half* buffer,
        int batch_size,
        int seq_len,
        int hidden_size,
        int cls_count,
        bool fast_fp32);

