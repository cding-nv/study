#pragma once
#if CUDART_VERSION >= 11040
#include "gemmLt.h"
#endif
#include "Add.h"
#include "Gelu.h"
#include "Dropout.h"
#include "Linear.h"
#include "utils.h"


class BertMlpTrain {
public:
    static int
    get_b_buff_size(int batch_seq,
                    int hidden_size,
                    double hidden_dropout_prob) {
        int b_buff_size = isActive(hidden_dropout_prob) ? batch_seq * hidden_size : 0;
        return b_buff_size;
    }

    template <typename T>
    static void
    forward(const T* input,
            const T* weight1,
            const T* bias1,
            const T* weight2,
            const T* bias2,
            uint8_t* dropout_mask,              // do_H_dropout
            T* gelu_inp,                        // w/o bias
            T* gelu_out,
            T* mlp_out,
            int64_t batch_seq,
            int64_t hidden_size,
            int64_t intermediate_size,
            double hidden_dropout_prob,
            cublasHandle_t handle,
            cudaStream_t stream) {
        const bool do_H_dropout = isActive(hidden_dropout_prob);
        const float H_dropout_scale = toScale(hidden_dropout_prob);

        Linear<T>::forward(
                input,
                weight1,
                gelu_inp,                       // media_numel, out
                handle,
                batch_seq,
                hidden_size,
                intermediate_size);

        GeluBias<T>::forward(
                gelu_inp,                       // media_numel, in
                bias1,
                gelu_out,                       // media_numel, out
                batch_seq,
                intermediate_size,
                stream);

        Linear<T>::forward(
                gelu_out,                       // media_numel, in
                weight2,
                mlp_out,                        // input_numel, out
                handle,
                batch_seq,
                intermediate_size,
                hidden_size);

        if (do_H_dropout) {
            DropoutBias<T>::forward(
                    bias2,
                    mlp_out,                    // input_numel, inout
                    dropout_mask,
                    hidden_dropout_prob,
                    H_dropout_scale,
                    batch_seq,
                    hidden_size,
                    stream);
        }
        else {
            AddBias<T>::forward(
                    bias2,
                    mlp_out,                    // input_numel, inout
                    batch_seq,
                    hidden_size,
                    stream);
        }
    }

    template <typename T>
    static void
    backward(T* buffer,                         // do_H_dropout
             const T* grad,
             const T* input,
             const T* weight1,
             const T* bias1,
             const T* weight2,
             const uint8_t* dropout_mask,       // do_H_dropout
             const T* gelu_inp,                 // w/o bias
             T* gelu_out,                       // backbuff
             T* grad_input,
             T* grad_weight1,
             T* grad_bias1,
             T* grad_weight2,
             T* grad_bias2,
             int64_t batch_seq,
             int64_t hidden_size,
             int64_t intermediate_size,
             double hidden_dropout_prob,
             cublasHandle_t handle,
             cudaStream_t stream) {
        const bool do_H_dropout = isActive(hidden_dropout_prob);
        const float H_dropout_scale = toScale(hidden_dropout_prob);

        T* buff_A = buffer;                     // input_numel
        T* buff_B = gelu_out;                   // media_numel

        if (do_H_dropout) {
            DropoutBias<T>::backward(
                    grad,
                    dropout_mask,
                    buff_A,                     // input_numel, out
                    grad_bias2,
                    H_dropout_scale,
                    batch_seq,
                    hidden_size,
                    stream);
        }
        else {
            AddBias<T>::backward(
                    grad,
                    grad_bias2,
                    batch_seq,
                    hidden_size,
                    stream);
        }

        Linear<T>::backward(
                do_H_dropout ? buff_A : grad,   // input_numel, in
                gelu_out,
                weight2,
                buff_B,                         // media_numel, out
                grad_weight2,
                handle,
                batch_seq,
                intermediate_size,
                hidden_size);

        GeluBias<T>::backward(
                buff_B,                         // media_numel, inout
                gelu_inp,
                bias1,
                grad_bias1,
                batch_seq,
                intermediate_size,
                stream);

        Linear<T>::backward(
                buff_B,                         // media_numel, in
                input,
                weight1,
                grad_input,
                grad_weight1,
                handle,
                batch_seq,
                hidden_size,
                intermediate_size);
    }
};


#if CUDART_VERSION >= 11040
class BertMlpTrainLt {
public:
    static int
    get_b_buff_size(int batch_seq,
                    int hidden_size,
                    double hidden_dropout_prob) {
        int b_buff_size = isActive(hidden_dropout_prob) ? batch_seq * hidden_size : 0;
        return b_buff_size;
    }

    template <typename T>
    static void
    forward(const T* input,
            const T* weight1,
            const T* bias1,
            const T* weight2,
            const T* bias2,
            uint8_t* dropout_mask,              // do_H_dropout
            T* gelu_inp,                        // w/ bias
            T* gelu_out,
            T* mlp_out,
            void* workspace,
            int64_t batch_seq,
            int64_t hidden_size,
            int64_t intermediate_size,
            double hidden_dropout_prob,
            cublasHandle_t handle,
            cudaStream_t stream) {
        const bool do_H_dropout = isActive(hidden_dropout_prob);
        const float H_dropout_scale = toScale(hidden_dropout_prob);

        gemmC_gelu_aux_bias(
                (cublasLtHandle_t)handle,
                stream,
                false,
                true,
                batch_seq,
                intermediate_size,
                hidden_size,
                input,
                weight1,
                bias1,
                gelu_inp,                       // media_numel, out
                gelu_out,                       // media_numel, out
                workspace);

        gemmC_bias(
                (cublasLtHandle_t)handle,
                stream,
                false,
                true,
                batch_seq,
                hidden_size,
                intermediate_size,
                gelu_out,                       // media_numel, in
                weight2,
                bias2,
                mlp_out,                        // input_numel, out
                workspace);

        if (do_H_dropout) {
            Dropout<T>::forward(
                    mlp_out,                    // input_numel, inout
                    dropout_mask,
                    hidden_dropout_prob,
                    H_dropout_scale,
                    batch_seq * hidden_size,
                    stream);
        }
    }

    template <typename T>
    static void
    backward(T* buffer,                         // do_H_dropout
             const T* grad,
             const T* input,
             const T* weight1,
             const T* weight2,
             const uint8_t* dropout_mask,       // do_H_dropout
             const T* gelu_inp,                 // w/ bias
             T* gelu_out,                       // backbuff
             T* grad_input,
             T* grad_weight1,
             T* grad_bias1,
             T* grad_weight2,
             T* grad_bias2,
             void* workspace,
             int64_t batch_seq,
             int64_t hidden_size,
             int64_t intermediate_size,
             double hidden_dropout_prob,
             cublasHandle_t handle,
             cudaStream_t stream) {
        const bool do_H_dropout = isActive(hidden_dropout_prob);
        const float H_dropout_scale = toScale(hidden_dropout_prob);

        T* buff_A = buffer;                     // input_numel
        T* buff_B = gelu_out;                   // media_numel

        if (do_H_dropout) {
            Dropout<T>::backward(
                    grad,
                    dropout_mask,
                    buff_A,                     // input_numel, out
                    H_dropout_scale,
                    batch_seq * hidden_size,
                    stream);
        }

        gemmC_bgradb(
                (cublasLtHandle_t)handle,
                stream,
                true,
                false,
                hidden_size,
                intermediate_size,
                batch_seq,
                do_H_dropout ? buff_A : grad,   // input_numel, in
                gelu_out,
                grad_bias2,
                grad_weight2,
                workspace);

        gemmC_dgelu_bgrad(
                (cublasLtHandle_t)handle,
                stream,
                false,
                false,
                batch_seq,
                intermediate_size,
                hidden_size,
                do_H_dropout ? buff_A : grad,   // input_numel, in
                weight2,
                grad_bias1,
                gelu_inp,
                buff_B,                         // media_numel, out
                workspace);

        Linear<T>::backward(
                buff_B,
                input,
                weight1,
                grad_input,
                grad_weight1,
                handle,
                batch_seq,
                hidden_size,
                intermediate_size);
    }
};
#endif

