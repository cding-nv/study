#pragma once
#if CUDART_VERSION >= 11040
#include "gemmLt.h"
#endif
#include "Add.h"
#include "Gelu.h"
#include "Linear.h"


class BertMlpInfer {
public:
    static int
    get_f_buff_size(int batch_seq,
                    int intermediate_size) {
        int f_buff_size = batch_seq * intermediate_size;
        return f_buff_size;
    }

    template <typename T>
    static void
    forward(const T* input,
            const T* weight1,
            const T* bias1,
            const T* weight2,
            const T* bias2,
            T* mlp_out,
            T* buffer,
            int64_t batch_seq,
            int64_t hidden_size,
            int64_t intermediate_size,
            cublasHandle_t handle,
            cudaStream_t stream) {
        Linear<T>::forward(
                input,
                weight1,
                buffer,                     // meida_numel, out
                handle,
                batch_seq,
                hidden_size,
                intermediate_size);

        GeluBias<T>::forward(
                bias1,
                buffer,                     // media_numel, inout
                batch_seq,
                intermediate_size,
                stream);

        Linear<T>::forward(
                buffer,                     // media_numel, in
                weight2,
                mlp_out,                    // input_numel, out
                handle,
                batch_seq,
                intermediate_size,
                hidden_size);

        AddBias<T>::forward(
                bias2,
                mlp_out,                    // input_numel, inout
                batch_seq,
                hidden_size,
                stream);
    }
};


#if CUDART_VERSION >= 11040
class BertMlpInferLt {
public:
    static int
    get_f_buff_size(int batch_seq,
                    int intermediate_size) {
        int f_buff_size = batch_seq * intermediate_size;
        return f_buff_size;
    }

    template <typename T>
    static void
    forward(const T* input,
            const T* weight1,
            const T* bias1,
            const T* weight2,
            const T* bias2,
            T* mlp_out,
            T* buffer,
            void* workspace,
            int64_t batch_seq,
            int64_t hidden_size,
            int64_t intermediate_size,
            cublasHandle_t handle,
            cudaStream_t stream) {
        gemmC_gelu_bias(
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
                buffer,                     // media_numel, out
                workspace);

        gemmC_bias(
                (cublasLtHandle_t)handle,
                stream,
                false,
                true,
                batch_seq,
                hidden_size,
                intermediate_size,
                buffer,                     // media_numel, in
                weight2,
                bias2,
                mlp_out,
                workspace);
    }
};
#endif

