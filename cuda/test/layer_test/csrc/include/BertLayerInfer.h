#pragma once
#include "Add.h"
#include "BatchGemm.h"
#include "BertCom.h"
#include "BiasedPermute.h"
#include "Gelu.h"
#include "LayerNorm.h"
#include "Linear.h"
#include "Permute.h"
#include "Softmax.h"


class BertLayerInfer {
private:
    template <typename T>
    static void
    pre_layernorm_forward(
            cudaStream_t stream,
            const T* hidden_states,
            const T* attention_mask,
            const T* normA_gamma,
            const T* normA_beta,
            const T* normB_gamma,
            const T* normB_beta,
            const T* linearA_weight,
            const T* linearA_bias,
            const T* linearB_weight,
            const T* linearB_bias,
            const T* linearC_weight,
            const T* linearC_bias,
            const T* linearD_weight,
            const T* linearD_bias,
            T* layer_out,
            T* buffer,
            int batch_size,
            int seq_len,
            int hidden_size,
            int num_attention_heads,
            int intermediate_size,
            float layer_norm_eps,
            cublasHandle_t handle) {
        int attention_head_size = hidden_size / num_attention_heads;
        int batch_seq = batch_size * seq_len;
        int input_numel = BertCom::get_input_numel(batch_size, seq_len, hidden_size);

        T* buff_A = layer_out;                              // input_numel
        T* buff_B = buffer + input_numel * 3;               // 3 * input_numel
        T* buff_C = buffer;                                 // 3 * input_numel
        T* buff_D = buffer + input_numel * 3;               // probs_numel
        T* buff_E = buffer;                                 // input_numel
        T* buff_F = buffer + input_numel * 2;               // media_numel
        T* buff_G = buffer + input_numel;                   // input_numel

        const T* buff_CQ = buff_C;                          // input_numel
        const T* buff_CK = buff_C + input_numel;            // input_numel
        const T* buff_CV = buff_C + input_numel * 2;        // input_numel

        /******************************************************************************\
                                   BertSelfAttention forward
        \******************************************************************************/

        // _normA
        LayerNorm<T>::forward(
                hidden_states,
                normA_gamma,
                normA_beta,
                buff_A,                         // A0, input_numel, out
                nullptr,                        // normA_rstd
                false,                          // training
                layer_norm_eps,
                batch_seq,
                hidden_size,
                stream);

        // _linearA
        Linear<T>::forward(
                buff_A,                         // A0, input_numel, in
                linearA_weight,
                buff_B,                         // 3 * input_numel, out
                handle,
                batch_seq,
                hidden_size,
                3 * hidden_size);

        BiasedPermute<T>::forward(
                buff_B,                         // 3 * input_numel, in
                linearA_bias,
                buff_C,                         // C0, 3 * input_numel, out
                batch_size,
                seq_len,
                hidden_size,
                num_attention_heads,
                attention_head_size,
                stream);

        // _gemm_qk
        BatchGemm<T, false, true>::forward(
                buff_CQ,                        // C0, input_numel, in
                buff_CK,                        // C0, input_numel, in
                buff_D,                         // probs_numel, out
                handle,
                batch_size * num_attention_heads,
                seq_len,
                seq_len,
                attention_head_size,
                1. / sqrt(attention_head_size));

        Softmax<T>::forwardM(
                attention_mask,
                buff_D,                         // probs_numel, inout
                batch_size,
                seq_len,
                num_attention_heads,
                stream);

        // _gemm_pv
        BatchGemm<T, false, false>::forward(
                buff_D,                         // probs_numel, in
                buff_CV,                        // C0, input_numel, in
                buff_A,                         // A1, input_numel, out
                handle,
                batch_size * num_attention_heads,
                seq_len,
                attention_head_size,
                seq_len);

        Permute<T>::forward(
                buff_A,                         // A1, input_numel, in
                buff_E,                         // E0, input_numel, out
                batch_size,
                seq_len,
                hidden_size,
                num_attention_heads,
                attention_head_size,
                stream);

        /******************************************************************************\
                                     BertSelfOutput forward
        \******************************************************************************/

        // _linearB
        Linear<T>::forward(
                buff_E,                         // E0, input_numel, in
                linearB_weight,
                buff_A,                         // A2, input_numel, out
                handle,
                batch_seq,
                hidden_size,
                hidden_size);

        // inp + bias + res
        // _addA
        AddBiasRes<T>::forward(
                buff_A,                         // A2, input_numel, in
                linearB_bias,
                hidden_states,
                buff_E,                         // E1, input_numel, out, hold on
                batch_seq,
                hidden_size,
                stream);

        /******************************************************************************\
                                    BertIntermediate forward
        \******************************************************************************/

        // _normB
        LayerNorm<T>::forward(
                buff_E,                         // E1, input_numel, in, hold on
                normB_gamma,
                normB_beta,
                buff_A,                         // A3, input_numel, out
                nullptr,                        // normB_rstd
                false,                          // training
                layer_norm_eps,
                batch_seq,
                hidden_size,
                stream);

        // _linearC
        Linear<T>::forward(
                buff_A,                         // A3, input_numel, in
                linearC_weight,
                buff_F,                         // media_numel, out
                handle,
                batch_seq,
                hidden_size,
                intermediate_size);

        GeluBias<T>::forward(
                linearC_bias,
                buff_F,                         // media_numel, inout
                batch_seq,
                intermediate_size,
                stream);

        /******************************************************************************\
                                       BertOutput forward
        \******************************************************************************/

        // _linearD
        Linear<T>::forward(
                buff_F,                         // media_numel, in
                linearD_weight,
                buff_G,                         // input_numel, out
                handle,
                batch_seq,
                intermediate_size,
                hidden_size);

        // inp + bias + res
        // _addB
        AddBiasRes<T>::forward(
                buff_G,                         // input_numel, in
                linearD_bias,
                buff_E,                         // E1, input_numel, in, hold off
                layer_out,
                batch_seq,
                hidden_size,
                stream);
    }

    template <typename T>
    static void
    post_layernorm_forward(
            cudaStream_t stream,
            const T* hidden_states,
            const T* attention_mask,
            const T* normA_gamma,
            const T* normA_beta,
            const T* normB_gamma,
            const T* normB_beta,
            const T* linearA_weight,
            const T* linearA_bias,
            const T* linearB_weight,
            const T* linearB_bias,
            const T* linearC_weight,
            const T* linearC_bias,
            const T* linearD_weight,
            const T* linearD_bias,
            T* layer_out,
            T* buffer,
            int batch_size,
            int seq_len,
            int hidden_size,
            int num_attention_heads,
            int intermediate_size,
            float layer_norm_eps,
            cublasHandle_t handle) {
        int attention_head_size = hidden_size / num_attention_heads;
        int batch_seq = batch_size * seq_len;
        int input_numel = BertCom::get_input_numel(batch_size, seq_len, hidden_size);

        T* buff_A = layer_out;                              // input_numel
        T* buff_B = buffer + input_numel * 3;               // 3 * input_numel
        T* buff_C = buffer;                                 // 3 * input_numel
        T* buff_D = buffer + input_numel * 3;               // probs_numel
        T* buff_E = buffer;                                 // input_numel
        T* buff_F = buffer + input_numel * 2;               // media_numel
        T* buff_G = buffer + input_numel;                   // input_numel

        const T* buff_CQ = buff_C;                          // input_numel
        const T* buff_CK = buff_C + input_numel;            // input_numel
        const T* buff_CV = buff_C + input_numel * 2;        // input_numel

        /******************************************************************************\
                                   BertSelfAttention forward
        \******************************************************************************/

        // _linearA
        Linear<T>::forward(
                hidden_states,
                linearA_weight,
                buff_B,                         // 3 * input_numel, out
                handle,
                batch_seq,
                hidden_size,
                3 * hidden_size);

        BiasedPermute<T>::forward(
                buff_B,                         // 3 * input_numel, in
                linearA_bias,
                buff_C,                         // C0, 3 * input_numel, out
                batch_size,
                seq_len,
                hidden_size,
                num_attention_heads,
                attention_head_size,
                stream);

        // _gemm_qk
        BatchGemm<T, false, true>::forward(
                buff_CQ,                        // C0, input_numel, in
                buff_CK,                        // C0, input_numel, in
                buff_D,                         // probs_numel, out
                handle,
                batch_size * num_attention_heads,
                seq_len,
                seq_len,
                attention_head_size,
                1. / sqrt(attention_head_size));

        Softmax<T>::forwardM(
                attention_mask,
                buff_D,                         // probs_numel, inout
                batch_size,
                seq_len,
                num_attention_heads,
                stream);

        // _gemm_pv
        BatchGemm<T, false, false>::forward(
                buff_D,                         // probs_numel, in
                buff_CV,                        // C0, input_numel, in
                buff_A,                         // A0, input_numel, out
                handle,
                batch_size * num_attention_heads,
                seq_len,
                attention_head_size,
                seq_len);

        Permute<T>::forward(
                buff_A,                         // A0, input_numel, in
                buff_E,                         // E0, input_numel, out
                batch_size,
                seq_len,
                hidden_size,
                num_attention_heads,
                attention_head_size,
                stream);

        /******************************************************************************\
                                     BertSelfOutput forward
        \******************************************************************************/

        // _linearB
        Linear<T>::forward(
                buff_E,                         // E0, input_numel, in
                linearB_weight,
                buff_A,                         // A1, input_numel, out
                handle,
                batch_seq,
                hidden_size,
                hidden_size);

        // layernorm(inp + bias + res)
        // _normA
        LayerNorm<T>::forwardB(
                buff_A,                         // A1, input_numel, in
                linearB_bias,
                hidden_states,
                normA_gamma,
                normA_beta,
                buff_E,                         // E1, input_numel, out, hold on
                nullptr,                        // normA_rstd
                false,                          // training
                layer_norm_eps,
                batch_seq,
                hidden_size,
                stream);

        /******************************************************************************\
                                    BertIntermediate forward
        \******************************************************************************/

        // _linearC
        Linear<T>::forward(
                buff_E,                         // E1, input_numel, in, hold on
                linearC_weight,
                buff_F,                         // media_numel, out
                handle,
                batch_seq,
                hidden_size,
                intermediate_size);

        GeluBias<T>::forward(
                linearC_bias,
                buff_F,                         // media_numel, inout
                batch_seq,
                intermediate_size,
                stream);

        /******************************************************************************\
                                       BertOutput forward
        \******************************************************************************/

        // _linearD
        Linear<T>::forward(
                buff_F,                         // media_numel, in
                linearD_weight,
                buff_G,                         // input_numel, out
                handle,
                batch_seq,
                intermediate_size,
                hidden_size);

        // layernorm(inp + bias + res)
        // _normB
        LayerNorm<T>::forwardB(
                buff_G,                         // input_numel, in
                linearD_bias,
                buff_E,                         // E1, input_numel, in, hold off
                normB_gamma,
                normB_beta,
                layer_out,
                nullptr,                        // normB_rstd
                false,                          // training
                layer_norm_eps,
                batch_seq,
                hidden_size,
                stream);
    }
public:
    static int
    get_f_buff_size(int batch_size,
                    int seq_len,
                    int hidden_size,
                    int num_attention_heads,
                    int intermediate_size) {
        int input_numel = BertCom::get_input_numel(batch_size, seq_len, hidden_size);
        int probs_numel = BertCom::get_probs_numel(batch_size, seq_len, num_attention_heads);
        int media_numel = BertCom::get_media_numel(batch_size, seq_len, intermediate_size);
        int f_buff_size = 2 * input_numel + max(media_numel, input_numel + max(probs_numel, 3 * input_numel));
        return f_buff_size;
    }

    template <typename T>
    static void
    forward(cudaStream_t stream,
            const T* hidden_states,
            const T* attention_mask,
            const T* normA_gamma,
            const T* normA_beta,
            const T* normB_gamma,
            const T* normB_beta,
            const T* linearA_weight,
            const T* linearA_bias,
            const T* linearB_weight,
            const T* linearB_bias,
            const T* linearC_weight,
            const T* linearC_bias,
            const T* linearD_weight,
            const T* linearD_bias,
            T* layer_out,
            T* buffer,
            int batch_size,
            int seq_len,
            int hidden_size,
            bool pre_layernorm,
            int num_attention_heads,
            int intermediate_size,
            float layer_norm_eps,
            cublasHandle_t handle) {
        auto forward_func = pre_layernorm ? pre_layernorm_forward<T> : post_layernorm_forward<T>;
        forward_func(
                stream,
                hidden_states,
                attention_mask,
                normA_gamma,
                normA_beta,
                normB_gamma,
                normB_beta,
                linearA_weight,
                linearA_bias,
                linearB_weight,
                linearB_bias,
                linearC_weight,
                linearC_bias,
                linearD_weight,
                linearD_bias,
                layer_out,
                buffer,
                batch_size,
                seq_len,
                hidden_size,
                num_attention_heads,
                intermediate_size,
                layer_norm_eps,
                handle);
    }
};

