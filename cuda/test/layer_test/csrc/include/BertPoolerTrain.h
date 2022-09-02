#pragma once
#include "BiasedTanh.h"
#include "Context.h"
#include "Linear.h"
#include "SliceSqueeze.h"


class BertPoolerTrain {
public:
    template <typename T>
    static void
    forward(const T* hidden_states,
            const T* linear_weight,
            const T* linear_bias,
            T* cls_token,
            T* pooled_output,
            int64_t batch_size,
            int64_t seq_len,
            int64_t hidden_size,
            int64_t cls_count,
            cudaStream_t stream) {
        int64_t batch_cnt = batch_size * cls_count;
        cublasHandle_t handle = Context::instance().getHandle(stream);

        SliceSqueeze<T>::forward(
                hidden_states,
                cls_token,
                batch_size,
                seq_len,
                hidden_size,
                cls_count,
                stream);

        Linear<T>::forward(
                cls_token,
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

    template <typename T>
    static void
    backward(const T* grad,
             const T* linear_weight,
             T* cls_token,                  // in, buff
             const T* pooled_output,
             T* grad_hidden_states,         // out, buff
             T* grad_linear_weight,
             T* grad_linear_bias,
             int64_t batch_size,
             int64_t seq_len,
             int64_t hidden_size,
             int64_t cls_count,
             cudaStream_t stream) {
        int64_t batch_cnt = batch_size * cls_count;
        cublasHandle_t handle = Context::instance().getHandle(stream);

        T* buff_A = grad_hidden_states;     // output_numel
        T* buff_B = cls_token;              // output_numel

        BiasedTanh<T>::backward(
                grad,
                pooled_output,
                buff_A,                     // output_numel, out
                grad_linear_bias,
                batch_cnt,
                hidden_size,
                stream);

        Linear<T>::backward(
                buff_A,                     // output_numel, in
                cls_token,
                linear_weight,
                buff_B,                     // output_numel, out
                grad_linear_weight,
                handle,
                batch_cnt,
                hidden_size,
                hidden_size);

        SliceSqueeze<T>::backward(
                buff_B,                     // output_numel, in
                grad_hidden_states,
                batch_size,
                seq_len,
                hidden_size,
                cls_count,
                stream);
    }
};

