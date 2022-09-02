#pragma once
#include "Add.h"
#include "BatchGemm.h"
#include "BertCom.h"
#include "BiasedPermute.h"
#include "DropPath.h"
#include "Dropout.h"
#include "Gelu.h"
#include "LayerNorm.h"
#include "Linear.h"
#include "Permute.h"
#include "Softmax.h"


class BertLayerTrain: public BertCom {
private:
    template <typename T>
    static void
    pre_layernorm_forward(
            T* buffer,
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
            T* qkv_layer,
            T* softmax_out,
            T* dropout_out,                     // do_P_dropout, backbuff
            uint8_t* dropout_mask,              // do_P_dropout
            uint8_t* dropoutA_mask,             // do_H_dropout
            uint8_t* dropoutB_mask,             // do_H_dropout
            uint8_t* dropPathA_mask,            // do_drop_path
            uint8_t* dropPathB_mask,            // do_drop_path
            T* normA_out,
            T* normA_rstd,
            T* normB_out,
            T* normB_rstd,
            T* context_layer,
            T* gelu_inp,
            T* gelu_out,
            T* layer_out,
            int64_t batch_size,
            int64_t seq_len,
            int64_t hidden_size,
            int64_t num_attention_heads,
            int64_t intermediate_size,
            double attention_probs_dropout_prob,
            double hidden_dropout_prob,
            double drop_path_prob,
            float layer_norm_eps,
            cublasHandle_t handle,
            cudaStream_t stream) {
        int64_t attention_head_size = hidden_size / num_attention_heads;
        int64_t batch_seq = batch_size * seq_len;
        int64_t input_numel = get_input_numel(batch_size, seq_len, hidden_size);
        bool do_H_dropout = isActive(hidden_dropout_prob);
        bool do_drop_path = isActive(drop_path_prob);
        float H_dropout_scale = toScale(hidden_dropout_prob);
        float drop_path_scale = toScale(drop_path_prob);
        int8_t hidden_drop_opt = (do_H_dropout << 1) + do_drop_path;

        bool reusable = is_reusable(hidden_size, intermediate_size);
        T* buff_A = reusable ? gelu_inp : buffer;           // 3 * input_numel
        T* buff_B = buffer;                                 // input_numel
        T* buff_C = buffer + input_numel;                   // input_numel

        const T* q_layer = qkv_layer;                       // input_numel
        const T* k_layer = qkv_layer + input_numel;         // input_numel
        const T* v_layer = qkv_layer + input_numel * 2;     // input_numel

        /******************************************************************************\
                                   BertSelfAttention forward
        \******************************************************************************/

        // _normA
        LayerNorm<T>::forward(
                hidden_states,
                normA_gamma,
                normA_beta,
                normA_out,
                normA_rstd,
                true,                           // training
                layer_norm_eps,
                batch_seq,
                hidden_size,
                stream);

        // _linearA
        Linear<T>::forward(
                normA_out,
                linearA_weight,
                buff_A,                         // 3 * input_numel, out
                handle,
                batch_seq,
                hidden_size,
                3 * hidden_size);

        BiasedPermute<T>::forward(
                buff_A,                         // 3 * input_numel, in
                linearA_bias,
                qkv_layer,
                batch_size,
                seq_len,
                hidden_size,
                num_attention_heads,
                attention_head_size,
                stream);

        // _gemm_qk
        BatchGemm<T, false, true>::forward(
                q_layer,
                k_layer,
                softmax_out,
                handle,
                batch_size * num_attention_heads,
                seq_len,
                seq_len,
                attention_head_size,
                1. / sqrt(attention_head_size));

        Softmax<T>::forwardM(
                attention_mask,
                softmax_out,
                batch_size,
                seq_len,
                num_attention_heads,
                stream);

        if (isActive(attention_probs_dropout_prob)) {
            Dropout<T>::forward(
                    softmax_out,
                    dropout_out,
                    dropout_mask,
                    attention_probs_dropout_prob,
                    toScale(attention_probs_dropout_prob),
                    get_probs_numel(batch_size, seq_len, num_attention_heads),
                    stream);

            // _gemm_pv
            BatchGemm<T, false, false>::forward(
                    dropout_out,
                    v_layer,
                    buff_B,                     // B0, input_numel, out
                    handle,
                    batch_size * num_attention_heads,
                    seq_len,
                    attention_head_size,
                    seq_len);
        }
        else {
            // _gemm_pv
            BatchGemm<T, false, false>::forward(
                    softmax_out,
                    v_layer,
                    buff_B,                     // B0, input_numel, out
                    handle,
                    batch_size * num_attention_heads,
                    seq_len,
                    attention_head_size,
                    seq_len);
        }

        Permute<T>::forward(
                buff_B,                         // B0, input_numel, in
                context_layer,
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
                context_layer,
                linearB_weight,
                buff_B,                         // B1, input_numel, out
                handle,
                batch_seq,
                hidden_size,
                hidden_size);

        switch (hidden_drop_opt) {
            case 3:
                // dropout(inp + bias)
                // _dropoutA
                DropoutBias<T>::forward(
                        buff_B,                 // B1, input_numel, in
                        linearB_bias,
                        buff_C,                 // input_numel, out
                        dropoutA_mask,
                        hidden_dropout_prob,
                        H_dropout_scale,
                        batch_seq,
                        hidden_size,
                        stream);

                // drop_path(inp) + res
                // _dropPathA
                DropPathRes<T>::forward(
                        buff_C,                 // input_numel, inout, hold on
                        hidden_states,
                        dropPathA_mask,
                        drop_path_prob,
                        drop_path_scale,
                        batch_size,
                        seq_len,
                        hidden_size,
                        stream);
                break;
            case 2:
                // dropout(inp + bias) + res
                // _dropoutA
                DropoutBiasRes<T>::forward(
                        buff_B,                 // B1, input_numel, in
                        linearB_bias,
                        hidden_states,
                        buff_C,                 // input_numel, out, hold on
                        dropoutA_mask,
                        hidden_dropout_prob,
                        H_dropout_scale,
                        batch_seq,
                        hidden_size,
                        stream);
                break;
            case 1:
                // drop_path(inp + bias) + res
                // _dropPathA
                DropPathBiasRes<T>::forward(
                        buff_B,                 // B1, input_numel, in
                        linearB_bias,
                        hidden_states,
                        buff_C,                 // input_numel, out, hold on
                        dropPathA_mask,
                        drop_path_prob,
                        drop_path_scale,
                        batch_size,
                        seq_len,
                        hidden_size,
                        stream);
                break;
            case 0:
                // inp + bias + res
                // _addA
                AddBiasRes<T>::forward(
                        buff_B,                 // B1, input_numel, in
                        linearB_bias,
                        hidden_states,
                        buff_C,                 // input_numel, out, hold on
                        batch_seq,
                        hidden_size,
                        stream);
                break;
        }

        /******************************************************************************\
                                    BertIntermediate forward
        \******************************************************************************/

        // _normB
        LayerNorm<T>::forward(
                buff_C,                         // input_numel, in, hold on
                normB_gamma,
                normB_beta,
                normB_out,
                normB_rstd,
                true,                           // training
                layer_norm_eps,
                batch_seq,
                hidden_size,
                stream);

        // _linearC
        Linear<T>::forward(
                normB_out,
                linearC_weight,
                gelu_inp,
                handle,
                batch_seq,
                hidden_size,
                intermediate_size);

        GeluBias<T>::forward(
                gelu_inp,
                linearC_bias,
                gelu_out,
                batch_seq,
                intermediate_size,
                stream);

        /******************************************************************************\
                                       BertOutput forward
        \******************************************************************************/

        // _linearD
        Linear<T>::forward(
                gelu_out,
                linearD_weight,
                buff_B,                         // B2, input_numel, out
                handle,
                batch_seq,
                intermediate_size,
                hidden_size);

        switch (hidden_drop_opt) {
            case 3:
                // dropout(inp + bias)
                // _dropoutB
                DropoutBias<T>::forward(
                        buff_B,                 // B2, input_numel, in
                        linearD_bias,
                        layer_out,
                        dropoutB_mask,
                        hidden_dropout_prob,
                        H_dropout_scale,
                        batch_seq,
                        hidden_size,
                        stream);

                // drop_path(inp) + res
                // _dropPathB
                DropPathRes<T>::forward(
                        layer_out,
                        buff_C,                 // input_numel, in, hold off
                        dropPathB_mask,
                        drop_path_prob,
                        drop_path_scale,
                        batch_size,
                        seq_len,
                        hidden_size,
                        stream);
                break;
            case 2:
                // dropout(inp + bias) + res
                // _dropoutB
                DropoutBiasRes<T>::forward(
                        buff_B,                 // B2, input_numel, in
                        linearD_bias,
                        buff_C,                 // input_numel, in, hold off
                        layer_out,
                        dropoutB_mask,
                        hidden_dropout_prob,
                        H_dropout_scale,
                        batch_seq,
                        hidden_size,
                        stream);
                break;
            case 1:
                // drop_path(inp + bias) + res
                // _dropPathB
                DropPathBiasRes<T>::forward(
                        buff_B,                 // B2, input_numel, in
                        linearD_bias,
                        buff_C,                 // input_numel, in, hold off
                        layer_out,
                        dropPathB_mask,
                        drop_path_prob,
                        drop_path_scale,
                        batch_size,
                        seq_len,
                        hidden_size,
                        stream);
                break;
            case 0:
                // inp + bias + res
                // _addB
                AddBiasRes<T>::forward(
                        buff_B,                 // B2, input_numel, in
                        linearD_bias,
                        buff_C,                 // input_numel, in, hold off
                        layer_out,
                        batch_seq,
                        hidden_size,
                        stream);
                break;
        }
    }

    template <typename T>
    static void
    post_layernorm_forward(
            T* buffer,
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
            T* qkv_layer,
            T* softmax_out,
            T* dropout_out,                     // do_P_dropout, backbuff
            uint8_t* dropout_mask,              // do_P_dropout
            uint8_t* dropoutA_mask,             // do_H_dropout
            uint8_t* dropoutB_mask,             // do_H_dropout
            uint8_t* dropPathA_mask,            // do_drop_path
            uint8_t* dropPathB_mask,            // do_drop_path
            T* normA_out,
            T* normA_rstd,
            T* normB_out,                       // nullptr, backbuff
            T* normB_rstd,
            T* context_layer,
            T* gelu_inp,
            T* gelu_out,
            T* layer_out,
            int64_t batch_size,
            int64_t seq_len,
            int64_t hidden_size,
            int64_t num_attention_heads,
            int64_t intermediate_size,
            double attention_probs_dropout_prob,
            double hidden_dropout_prob,
            double drop_path_prob,
            float layer_norm_eps,
            cublasHandle_t handle,
            cudaStream_t stream) {
        int64_t attention_head_size = hidden_size / num_attention_heads;
        int64_t batch_seq = batch_size * seq_len;
        int64_t input_numel = get_input_numel(batch_size, seq_len, hidden_size);
        bool do_H_dropout = isActive(hidden_dropout_prob);
        bool do_drop_path = isActive(drop_path_prob);
        float H_dropout_scale = toScale(hidden_dropout_prob);
        float drop_path_scale = toScale(drop_path_prob);
        int8_t hidden_drop_opt = (do_H_dropout << 1) + do_drop_path;

        bool reusable = is_reusable(hidden_size, intermediate_size);
        T* buff_A = reusable ? gelu_inp : buffer;           // 3 * input_numel
        T* buff_B = buffer;                                 // input_numel
        T* buff_C = buffer + input_numel;                   // input_numel

        const T* q_layer = qkv_layer;                       // input_numel
        const T* k_layer = qkv_layer + input_numel;         // input_numel
        const T* v_layer = qkv_layer + input_numel * 2;     // input_numel

        /******************************************************************************\
                                   BertSelfAttention forward
        \******************************************************************************/

        // _linearA
        Linear<T>::forward(
                hidden_states,
                linearA_weight,
                buff_A,                         // 3 * input_numel, out
                handle,
                batch_seq,
                hidden_size,
                3 * hidden_size);

        BiasedPermute<T>::forward(
                buff_A,                         // 3 * input_numel, in
                linearA_bias,
                qkv_layer,
                batch_size,
                seq_len,
                hidden_size,
                num_attention_heads,
                attention_head_size,
                stream);

        // _gemm_qk
        BatchGemm<T, false, true>::forward(
                q_layer,
                k_layer,
                softmax_out,
                handle,
                batch_size * num_attention_heads,
                seq_len,
                seq_len,
                attention_head_size,
                1. / sqrt(attention_head_size));

        Softmax<T>::forwardM(
                attention_mask,
                softmax_out,
                batch_size,
                seq_len,
                num_attention_heads,
                stream);

        if (isActive(attention_probs_dropout_prob)) {
            Dropout<T>::forward(
                    softmax_out,
                    dropout_out,
                    dropout_mask,
                    attention_probs_dropout_prob,
                    toScale(attention_probs_dropout_prob),
                    get_probs_numel(batch_size, seq_len, num_attention_heads),
                    stream);

            // _gemm_pv
            BatchGemm<T, false, false>::forward(
                    dropout_out,
                    v_layer,
                    buff_B,                     // B0, input_numel, out
                    handle,
                    batch_size * num_attention_heads,
                    seq_len,
                    attention_head_size,
                    seq_len);
        }
        else {
            // _gemm_pv
            BatchGemm<T, false, false>::forward(
                    softmax_out,
                    v_layer,
                    buff_B,                     // B0, input_numel, out
                    handle,
                    batch_size * num_attention_heads,
                    seq_len,
                    attention_head_size,
                    seq_len);
        }

        Permute<T>::forward(
                buff_B,                         // B0, input_numel, in
                context_layer,
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
                context_layer,
                linearB_weight,
                buff_B,                         // B1, input_numel, out
                handle,
                batch_seq,
                hidden_size,
                hidden_size);

        switch (hidden_drop_opt) {
            case 3:
                // dropout(inp + bias)
                // _dropoutA
                DropoutBias<T>::forward(
                        buff_B,                 // B1, input_numel, in
                        linearB_bias,
                        buff_C,                 // C0, input_numel, out
                        dropoutA_mask,
                        hidden_dropout_prob,
                        H_dropout_scale,
                        batch_seq,
                        hidden_size,
                        stream);

                // drop_path(inp) + res
                // _dropPathA
                DropPathRes<T>::forward(
                        buff_C,                 // C0, input_numel, inout
                        hidden_states,
                        dropPathA_mask,
                        drop_path_prob,
                        drop_path_scale,
                        batch_size,
                        seq_len,
                        hidden_size,
                        stream);

                // _normA
                LayerNorm<T>::forward(
                        buff_C,                 // C0, input_numel, in
                        normA_gamma,
                        normA_beta,
                        normA_out,
                        normA_rstd,
                        true,                   // training
                        layer_norm_eps,
                        batch_seq,
                        hidden_size,
                        stream);
                break;
            case 2:
                // dropout(inp + bias) + res
                // _dropoutA
                DropoutBiasRes<T>::forward(
                        buff_B,                 // B1, input_numel, in
                        linearB_bias,
                        hidden_states,
                        buff_C,                 // C0, input_numel, out
                        dropoutA_mask,
                        hidden_dropout_prob,
                        H_dropout_scale,
                        batch_seq,
                        hidden_size,
                        stream);

                // _normA
                LayerNorm<T>::forward(
                        buff_C,                 // C0, input_numel, in
                        normA_gamma,
                        normA_beta,
                        normA_out,
                        normA_rstd,
                        true,                   // training
                        layer_norm_eps,
                        batch_seq,
                        hidden_size,
                        stream);
                break;
            case 1:
                // drop_path(inp + bias) + res
                // _dropPathA
                DropPathBiasRes<T>::forward(
                        buff_B,                 // B1, input_numel, in
                        linearB_bias,
                        hidden_states,
                        buff_C,                 // C0, input_numel, out
                        dropPathA_mask,
                        drop_path_prob,
                        drop_path_scale,
                        batch_size,
                        seq_len,
                        hidden_size,
                        stream);

                // _normA
                LayerNorm<T>::forward(
                        buff_C,                 // C0, input_numel, in
                        normA_gamma,
                        normA_beta,
                        normA_out,
                        normA_rstd,
                        true,                   // training
                        layer_norm_eps,
                        batch_seq,
                        hidden_size,
                        stream);
                break;
            case 0:
                // layernorm(inp + bias + res)
                // _normA
                LayerNorm<T>::forwardB(
                        buff_B,                 // B1, input_numel, in
                        linearB_bias,
                        hidden_states,
                        normA_gamma,
                        normA_beta,
                        normA_out,
                        normA_rstd,
                        true,                   // training
                        layer_norm_eps,
                        batch_seq,
                        hidden_size,
                        stream);
                break;
        }

        /******************************************************************************\
                                    BertIntermediate forward
        \******************************************************************************/

        // _linearC
        Linear<T>::forward(
                normA_out,
                linearC_weight,
                gelu_inp,
                handle,
                batch_seq,
                hidden_size,
                intermediate_size);

        GeluBias<T>::forward(
                gelu_inp,
                linearC_bias,
                gelu_out,
                batch_seq,
                intermediate_size,
                stream);

        /******************************************************************************\
                                       BertOutput forward
        \******************************************************************************/

        // _linearD
        Linear<T>::forward(
                gelu_out,
                linearD_weight,
                buff_B,                         // B2, input_numel, out
                handle,
                batch_seq,
                intermediate_size,
                hidden_size);

        switch (hidden_drop_opt) {
            case 3:
                // dropout(inp + bias)
                // _dropoutB
                DropoutBias<T>::forward(
                        buff_B,                 // B2, input_numel, in
                        linearD_bias,
                        buff_C,                 // C1, input_numel, out
                        dropoutB_mask,
                        hidden_dropout_prob,
                        H_dropout_scale,
                        batch_seq,
                        hidden_size,
                        stream);

                // drop_path(inp) + res
                // _dropPathB
                DropPathRes<T>::forward(
                        buff_C,                 // C1, input_numel, inout
                        normA_out,
                        dropPathB_mask,
                        drop_path_prob,
                        drop_path_scale,
                        batch_size,
                        seq_len,
                        hidden_size,
                        stream);

                // _normB
                LayerNorm<T>::forward(
                        buff_C,                 // C1, input_numel, in
                        normB_gamma,
                        normB_beta,
                        layer_out,
                        normB_rstd,
                        true,                   // training
                        layer_norm_eps,
                        batch_seq,
                        hidden_size,
                        stream);
                break;
            case 2:
                // dropout(inp + bias) + res
                // _dropoutB
                DropoutBiasRes<T>::forward(
                        buff_B,                 // B2, input_numel, in
                        linearD_bias,
                        normA_out,
                        buff_C,                 // C1, input_numel, out
                        dropoutB_mask,
                        hidden_dropout_prob,
                        H_dropout_scale,
                        batch_seq,
                        hidden_size,
                        stream);

                // _normB
                LayerNorm<T>::forward(
                        buff_C,                 // C1, input_numel, in
                        normB_gamma,
                        normB_beta,
                        layer_out,
                        normB_rstd,
                        true,                   // training
                        layer_norm_eps,
                        batch_seq,
                        hidden_size,
                        stream);
                break;
            case 1:
                // drop_path(inp + bias) + res
                // _dropPathB
                DropPathBiasRes<T>::forward(
                        buff_B,                 // B2, input_numel, in
                        linearD_bias,
                        normA_out,
                        buff_C,                 // C1, input_numel, out
                        dropPathB_mask,
                        drop_path_prob,
                        drop_path_scale,
                        batch_size,
                        seq_len,
                        hidden_size,
                        stream);

                // _normB
                LayerNorm<T>::forward(
                        buff_C,                 // C1, input_numel, in
                        normB_gamma,
                        normB_beta,
                        layer_out,
                        normB_rstd,
                        true,                   // training
                        layer_norm_eps,
                        batch_seq,
                        hidden_size,
                        stream);
                break;
            case 0:
                // layernorm(inp + bias + res)
                // _normB
                LayerNorm<T>::forwardB(
                        buff_B,                 // B2, input_numel, in
                        linearD_bias,
                        normA_out,
                        normB_gamma,
                        normB_beta,
                        layer_out,
                        normB_rstd,
                        true,                   // training
                        layer_norm_eps,
                        batch_seq,
                        hidden_size,
                        stream);
                break;
        }
    }

    template <typename T>
    static void
    pre_layernorm_backward(
            T* buffer,
            const T* grad,
            const T* hidden_states,             // nullptr
            const T* normA_gamma,
            const T* normA_beta,
            const T* normB_gamma,
            const T* normB_beta,
            const T* linearA_weight,
            const T* linearB_weight,
            const T* linearC_weight,
            const T* linearC_bias,
            const T* linearD_weight,
            T* qkv_layer,                       // in, buff
            const T* softmax_out,
            T* dropout_out,                     // in, buff
            const uint8_t* dropout_mask,        // do_P_dropout
            const uint8_t* dropoutA_mask,       // do_H_dropout
            const uint8_t* dropoutB_mask,       // do_H_dropout
            const uint8_t* dropPathA_mask,      // do_drop_path
            const uint8_t* dropPathB_mask,      // do_drop_path
            const T* normA_out,
            const T* normA_rstd,
            T* normB_out,                       // in, buff
            const T* normB_rstd,
            T* context_layer,                   // in, buff
            T* gelu_inp,                        // in, buff
            T* gelu_out,                        // in, buff
            const T* layer_out,                 // nullptr
            T* grad_hidden_states,              // out, buff
            T* grad_normA_gamma,
            T* grad_normA_beta,
            T* grad_normB_gamma,
            T* grad_normB_beta,
            T* grad_linearA_weight,
            T* grad_linearA_bias,
            T* grad_linearB_weight,
            T* grad_linearB_bias,
            T* grad_linearC_weight,
            T* grad_linearC_bias,
            T* grad_linearD_weight,
            T* grad_linearD_bias,
            int64_t batch_size,
            int64_t seq_len,
            int64_t hidden_size,
            int64_t num_attention_heads,
            int64_t intermediate_size,
            double attention_probs_dropout_prob,
            double hidden_dropout_prob,
            double drop_path_prob,
            cublasHandle_t handle,
            cudaStream_t stream) {
        int64_t attention_head_size = hidden_size / num_attention_heads;
        int64_t batch_seq = batch_size * seq_len;
        int64_t input_numel = get_input_numel(batch_size, seq_len, hidden_size);
        bool do_H_dropout = isActive(hidden_dropout_prob);
        bool do_drop_path = isActive(drop_path_prob);
        float H_dropout_scale = toScale(hidden_dropout_prob);
        float drop_path_scale = toScale(drop_path_prob);
        int8_t hidden_drop_opt = (do_H_dropout << 1) + do_drop_path;

        bool reusable = is_reusable(hidden_size, intermediate_size);
        T* buff_A = grad_hidden_states;                     // input_numel
        T* buff_B = gelu_out;                               // media_numel
        T* buff_C = reusable ? gelu_inp : buffer;           // 3 * input_numel
        T* buff_D = normB_out;                              // input_numel
        T* buff_E = context_layer;                          // input_numel
        T* buff_F = dropout_out;                            // probs_numel
        T* buff_G = qkv_layer;                              // 3 * input_numel

        T* buff_CQ = buff_C;                                // input_numel
        T* buff_CK = buff_C + input_numel;                  // input_numel
        T* buff_CV = buff_C + input_numel * 2;              // input_numel
        const T* q_layer = qkv_layer;                       // input_numel
        const T* k_layer = qkv_layer + input_numel;         // input_numel
        const T* v_layer = qkv_layer + input_numel * 2;     // input_numel

        /******************************************************************************\
                                      BertOutput backward
        \******************************************************************************/

        switch (hidden_drop_opt) {
            case 3:
                // drop_path(inp) + res
                // _dropPathB
                DropPathRes<T>::backward(
                        grad,                   // in, hold on
                        dropPathB_mask,
                        buff_A,                 // input_numel, out
                        drop_path_scale,
                        batch_size,
                        seq_len,
                        hidden_size,
                        stream);

                // dropout(inp + bias)
                // _dropoutB
                DropoutBias<T>::backward(
                        buff_A,                 // input_numel, inout
                        dropoutB_mask,
                        grad_linearD_bias,
                        H_dropout_scale,
                        batch_seq,
                        hidden_size,
                        stream);
                break;
            case 2:
                // dropout(inp + bias) + res
                // _dropoutB
                DropoutBiasRes<T>::backward(
                        grad,                   // in, hold on
                        dropoutB_mask,
                        buff_A,                 // input_numel, out
                        grad_linearD_bias,
                        H_dropout_scale,
                        batch_seq,
                        hidden_size,
                        stream);
                break;
            case 1:
                // drop_path(inp + bias) + res
                // _dropPathB
                DropPathBiasRes<T>::backward(
                        grad,                   // in, hold on
                        dropPathB_mask,
                        buff_A,                 // input_numel, out
                        grad_linearD_bias,
                        drop_path_scale,
                        batch_size,
                        seq_len,
                        hidden_size,
                        stream);
                break;
            case 0:
                // inp + bias + res
                // _addB
                AddBiasRes<T>::backward(
                        grad,                   // in, hold on
                        buff_A,                 // input_numel, out
                        grad_linearD_bias,
                        batch_seq,
                        hidden_size,
                        stream);
                break;
        }

        // _linearD
        Linear<T>::backward(
                buff_A,                         // input_numel, in
                gelu_out,
                linearD_weight,
                buff_B,                         // media_numel, out
                grad_linearD_weight,
                handle,
                batch_seq,
                intermediate_size,
                hidden_size);

        /******************************************************************************\
                                   BertIntermediate backward
        \******************************************************************************/

        GeluBias<T>::backward(
                buff_B,                         // media_numel, inout
                gelu_inp,
                linearC_bias,
                grad_linearC_bias,
                batch_seq,
                intermediate_size,
                stream);

        // _linearC
        Linear<T>::backward(
                buff_B,                         // media_numel, in
                normB_out,
                linearC_weight,
                buff_C,                         // C0, input_numel, out
                grad_linearC_weight,
                handle,
                batch_seq,
                hidden_size,
                intermediate_size);

        // _normB
        LayerNorm<T>::backwardA(
                buff_C,                         // C0, input_numel, in
                grad,                           // hold off
                normB_out,
                normB_gamma,
                normB_beta,
                normB_rstd,
                grad_normB_gamma,
                grad_normB_beta,
                grad_hidden_states,
                batch_seq,
                hidden_size,
                stream);

        /******************************************************************************\
                                    BertSelfOutput backward
        \******************************************************************************/

        switch (hidden_drop_opt) {
            case 3:
                // drop_path(inp) + res
                // _dropPathA
                DropPathRes<T>::backward(
                        grad_hidden_states,
                        dropPathA_mask,
                        buff_D,                 // D0, input_numel, out
                        drop_path_scale,
                        batch_size,
                        seq_len,
                        hidden_size,
                        stream);

                // dropout(inp + bias)
                // _dropoutA
                DropoutBias<T>::backward(
                        buff_D,                 // D0, input_numel, inout
                        dropoutA_mask,
                        grad_linearB_bias,
                        H_dropout_scale,
                        batch_seq,
                        hidden_size,
                        stream);
                break;
            case 2:
                // dropout(inp + bias) + res
                // _dropoutA
                DropoutBiasRes<T>::backward(
                        grad_hidden_states,
                        dropoutA_mask,
                        buff_D,                 // D0, input_numel, out
                        grad_linearB_bias,
                        H_dropout_scale,
                        batch_seq,
                        hidden_size,
                        stream);
                break;
            case 1:
                // drop_path(inp + bias) + res
                // _dropPathA
                DropPathBiasRes<T>::backward(
                        grad_hidden_states,
                        dropPathA_mask,
                        buff_D,                 // D0, input_numel, out
                        grad_linearB_bias,
                        drop_path_scale,
                        batch_size,
                        seq_len,
                        hidden_size,
                        stream);
                break;
            case 0:
                // inp + bias + res
                // _addA
                AddBiasRes<T>::backward(
                        grad_hidden_states,
                        buff_D,                 // D0, input_numel, out
                        grad_linearB_bias,
                        batch_seq,
                        hidden_size,
                        stream);
                break;
        }

        // _linearB
        Linear<T>::backward(
                buff_D,                         // D0, input_numel, in
                context_layer,
                linearB_weight,
                buff_E,                         // input_numel, out
                grad_linearB_weight,
                handle,
                batch_seq,
                hidden_size,
                hidden_size);

        /******************************************************************************\
                                   BertSelfAttention backward
        \******************************************************************************/

        Permute<T>::backward(
                buff_E,                         // input_numel, in
                buff_D,                         // D1, input_numel, out
                batch_size,
                seq_len,
                hidden_size,
                num_attention_heads,
                attention_head_size,
                stream);

        if (isActive(attention_probs_dropout_prob)) {
            // _gemm_pv
            BatchGemm<T, false, false>::backward(
                    buff_D,                     // D1, input_numel, in
                    dropout_out,
                    v_layer,
                    buff_F,                     // probs_numel, out
                    buff_CV,                    // C1, input_numel, out
                    handle,
                    batch_size * num_attention_heads,
                    seq_len,
                    attention_head_size,
                    seq_len);

            Dropout<T>::backward(
                    buff_F,                     // probs_numel, inout
                    dropout_mask,
                    toScale(attention_probs_dropout_prob),
                    get_probs_numel(batch_size, seq_len, num_attention_heads),
                    stream);
        }
        else {
            // _gemm_pv
            BatchGemm<T, false, false>::backward(
                    buff_D,                     // D1, input_numel, in
                    softmax_out,
                    v_layer,
                    buff_F,                     // probs_numel, out
                    buff_CV,                    // C1, input_numel, out
                    handle,
                    batch_size * num_attention_heads,
                    seq_len,
                    attention_head_size,
                    seq_len);
        }

        Softmax<T>::backward(
                softmax_out,
                buff_F,                         // probs_numel, inout
                batch_size * num_attention_heads * seq_len,
                seq_len,
                stream);

        // _gemm_qk
        BatchGemm<T, false, true>::backward(
                buff_F,                         // probs_numel, in
                q_layer,
                k_layer,
                buff_CQ,                        // C1, input_numel, out
                buff_CK,                        // C1, input_numel, out
                handle,
                batch_size * num_attention_heads,
                seq_len,
                seq_len,
                attention_head_size,
                1. / sqrt(attention_head_size));

        BiasedPermute<T>::backward(
                buff_C,                         // C1, 3 * input_numel, in
                buff_G,                         // 3 * input_numel, out
                grad_linearA_bias,
                batch_size,
                seq_len,
                hidden_size,
                num_attention_heads,
                attention_head_size,
                stream);

        // _linearA
        Linear<T>::backward(
                buff_G,                         // 3 * input_numel, in
                normA_out,
                linearA_weight,
                buff_D,                         // D2, input_numel, out
                grad_linearA_weight,
                handle,
                batch_seq,
                hidden_size,
                3 * hidden_size);

        // _normA
        LayerNorm<T>::backwardA(
                buff_D,                         // D2, input_numel, in
                normA_out,
                normA_gamma,
                normA_beta,
                normA_rstd,
                grad_normA_gamma,
                grad_normA_beta,
                grad_hidden_states,
                batch_seq,
                hidden_size,
                stream);
    }

    template <typename T>
    static void
    post_layernorm_backward(
            T* buffer,
            const T* grad,
            const T* hidden_states,
            const T* normA_gamma,
            const T* normA_beta,
            const T* normB_gamma,
            const T* normB_beta,
            const T* linearA_weight,
            const T* linearB_weight,
            const T* linearC_weight,
            const T* linearC_bias,
            const T* linearD_weight,
            T* qkv_layer,                       // in, buff
            const T* softmax_out,
            T* dropout_out,                     // in, buff
            const uint8_t* dropout_mask,        // do_P_dropout
            const uint8_t* dropoutA_mask,       // do_H_dropout
            const uint8_t* dropoutB_mask,       // do_H_dropout
            const uint8_t* dropPathA_mask,      // do_drop_path
            const uint8_t* dropPathB_mask,      // do_drop_path
            const T* normA_out,
            const T* normA_rstd,
            T* normB_out,                       // in, buff
            const T* normB_rstd,
            T* context_layer,                   // in, buff
            T* gelu_inp,                        // in, buff
            T* gelu_out,                        // in, buff
            const T* layer_out,
            T* grad_hidden_states,              // out, buff
            T* grad_normA_gamma,
            T* grad_normA_beta,
            T* grad_normB_gamma,
            T* grad_normB_beta,
            T* grad_linearA_weight,
            T* grad_linearA_bias,
            T* grad_linearB_weight,
            T* grad_linearB_bias,
            T* grad_linearC_weight,
            T* grad_linearC_bias,
            T* grad_linearD_weight,
            T* grad_linearD_bias,
            int64_t batch_size,
            int64_t seq_len,
            int64_t hidden_size,
            int64_t num_attention_heads,
            int64_t intermediate_size,
            double attention_probs_dropout_prob,
            double hidden_dropout_prob,
            double drop_path_prob,
            cublasHandle_t handle,
            cudaStream_t stream) {
        int64_t attention_head_size = hidden_size / num_attention_heads;
        int64_t batch_seq = batch_size * seq_len;
        int64_t input_numel = get_input_numel(batch_size, seq_len, hidden_size);
        bool do_H_dropout = isActive(hidden_dropout_prob);
        bool do_drop_path = isActive(drop_path_prob);
        float H_dropout_scale = toScale(hidden_dropout_prob);
        float drop_path_scale = toScale(drop_path_prob);
        int8_t hidden_drop_opt = (do_H_dropout << 1) + do_drop_path;

        bool reusable = is_reusable(hidden_size, intermediate_size);
        T* buff_A = grad_hidden_states;                     // input_numel
        T* buff_B = gelu_out;                               // media_numel
        T* buff_C = reusable ? gelu_inp : buffer;           // 3 * input_numel
        T* buff_D = normB_out;                              // input_numel
        T* buff_E = context_layer;                          // input_numel
        T* buff_F = dropout_out;                            // probs_numel
        T* buff_G = qkv_layer;                              // 3 * input_numel

        T* buff_CQ = buff_C;                                // input_numel
        T* buff_CK = buff_C + input_numel;                  // input_numel
        T* buff_CV = buff_C + input_numel * 2;              // input_numel
        const T* q_layer = qkv_layer;                       // input_numel
        const T* k_layer = qkv_layer + input_numel;         // input_numel
        const T* v_layer = qkv_layer + input_numel * 2;     // input_numel

        /******************************************************************************\
                                      BertOutput backward
        \******************************************************************************/

        switch (hidden_drop_opt) {
            case 3:
                // _normB
                LayerNorm<T>::backward(
                        grad,
                        layer_out,
                        normB_gamma,
                        normB_beta,
                        normB_rstd,
                        grad_normB_gamma,
                        grad_normB_beta,
                        buff_D,                 // D0, input_numel, out, hold on
                        batch_seq,
                        hidden_size,
                        stream);

                // drop_path(inp) + res
                // _dropPathB
                DropPathRes<T>::backward(
                        buff_D,                 // D0, input_numel, in, hold on
                        dropPathB_mask,
                        buff_A,                 // input_numel, out
                        drop_path_scale,
                        batch_size,
                        seq_len,
                        hidden_size,
                        stream);

                // dropout(inp + bias)
                // _dropoutB
                DropoutBias<T>::backward(
                        buff_A,                 // input_numel, inout
                        dropoutB_mask,
                        grad_linearD_bias,
                        H_dropout_scale,
                        batch_seq,
                        hidden_size,
                        stream);
                break;
            case 2:
                // _normB
                LayerNorm<T>::backward(
                        grad,
                        layer_out,
                        normB_gamma,
                        normB_beta,
                        normB_rstd,
                        grad_normB_gamma,
                        grad_normB_beta,
                        buff_D,                 // D0, input_numel, out, hold on
                        batch_seq,
                        hidden_size,
                        stream);

                // dropout(inp + bias) + res
                // _dropoutB
                DropoutBiasRes<T>::backward(
                        buff_D,                 // D0, input_numel, in, hold on
                        dropoutB_mask,
                        buff_A,                 // input_numel, out
                        grad_linearD_bias,
                        H_dropout_scale,
                        batch_seq,
                        hidden_size,
                        stream);
                break;
            case 1:
                // _normB
                LayerNorm<T>::backward(
                        grad,
                        layer_out,
                        normB_gamma,
                        normB_beta,
                        normB_rstd,
                        grad_normB_gamma,
                        grad_normB_beta,
                        buff_D,                 // D0, input_numel, out, hold on
                        batch_seq,
                        hidden_size,
                        stream);

                // drop_path(inp + bias) + res
                // _dropPathB
                DropPathBiasRes<T>::backward(
                        buff_D,                 // D0, input_numel, in, hold on
                        dropPathB_mask,
                        buff_A,                 // input_numel, out
                        grad_linearD_bias,
                        drop_path_scale,
                        batch_size,
                        seq_len,
                        hidden_size,
                        stream);
                break;
            case 0:
                // layernorm(inp + bias + res)
                // _normB
                LayerNorm<T>::backwardB(
                        grad,
                        layer_out,
                        normB_gamma,
                        normB_beta,
                        normB_rstd,
                        grad_normB_gamma,
                        grad_normB_beta,
                        buff_A,                 // input_numel, out
                        grad_linearD_bias,
                        buff_D,                 // D0, input_numel, out, hold on
                        batch_seq,
                        hidden_size,
                        stream);
                break;
        }

        // _linearD
        Linear<T>::backward(
                buff_A,                         // input_numel, in
                gelu_out,
                linearD_weight,
                buff_B,                         // media_numel, out
                grad_linearD_weight,
                handle,
                batch_seq,
                intermediate_size,
                hidden_size);

        /******************************************************************************\
                                   BertIntermediate backward
        \******************************************************************************/

        GeluBias<T>::backward(
                buff_B,                         // media_numel, inout
                gelu_inp,
                linearC_bias,
                grad_linearC_bias,
                batch_seq,
                intermediate_size,
                stream);

        // _linearC
        Linear<T>::backwardA(
                buff_B,                         // media_numel, in
                normA_out,
                linearC_weight,
                buff_D,                         // D0, input_numel, inout, hold on
                grad_linearC_weight,
                handle,
                batch_seq,
                hidden_size,
                intermediate_size);

        /******************************************************************************\
                                    BertSelfOutput backward
        \******************************************************************************/

        switch (hidden_drop_opt) {
            case 3:
                // _normA
                LayerNorm<T>::backward(
                        buff_D,                 // D0, input_numel, in, hold off
                        normA_out,
                        normA_gamma,
                        normA_beta,
                        normA_rstd,
                        grad_normA_gamma,
                        grad_normA_beta,
                        grad_hidden_states,
                        batch_seq,
                        hidden_size,
                        stream);

                // drop_path(inp) + res
                // _dropPathA
                DropPathRes<T>::backward(
                        grad_hidden_states,
                        dropPathA_mask,
                        buff_C,                 // C0, input_numel, out
                        drop_path_scale,
                        batch_size,
                        seq_len,
                        hidden_size,
                        stream);

                // dropout(inp + bias)
                // _dropoutA
                DropoutBias<T>::backward(
                        buff_C,                 // C0, input_numel, inout
                        dropoutA_mask,
                        grad_linearB_bias,
                        H_dropout_scale,
                        batch_seq,
                        hidden_size,
                        stream);
                break;
            case 2:
                // _normA
                LayerNorm<T>::backward(
                        buff_D,                 // D0, input_numel, in, hold off
                        normA_out,
                        normA_gamma,
                        normA_beta,
                        normA_rstd,
                        grad_normA_gamma,
                        grad_normA_beta,
                        grad_hidden_states,
                        batch_seq,
                        hidden_size,
                        stream);

                // dropout(inp + bias) + res
                // _dropoutA
                DropoutBiasRes<T>::backward(
                        grad_hidden_states,
                        dropoutA_mask,
                        buff_C,                 // C0, input_numel, out
                        grad_linearB_bias,
                        H_dropout_scale,
                        batch_seq,
                        hidden_size,
                        stream);
                break;
            case 1:
                // _normA
                LayerNorm<T>::backward(
                        buff_D,                 // D0, input_numel, in, hold off
                        normA_out,
                        normA_gamma,
                        normA_beta,
                        normA_rstd,
                        grad_normA_gamma,
                        grad_normA_beta,
                        grad_hidden_states,
                        batch_seq,
                        hidden_size,
                        stream);

                // drop_path(inp + bias) + res
                // _dropPathA
                DropPathBiasRes<T>::backward(
                        grad_hidden_states,
                        dropPathA_mask,
                        buff_C,                 // C0, input_numel, out
                        grad_linearB_bias,
                        drop_path_scale,
                        batch_size,
                        seq_len,
                        hidden_size,
                        stream);
                break;
            case 0:
                // layernorm(inp + bias + res)
                // _normA
                LayerNorm<T>::backwardB(
                        buff_D,                 // D0, input_numel, in, hold off
                        normA_out,
                        normA_gamma,
                        normA_beta,
                        normA_rstd,
                        grad_normA_gamma,
                        grad_normA_beta,
                        buff_C,                 // C0, input_numel, out
                        grad_linearB_bias,
                        grad_hidden_states,
                        batch_seq,
                        hidden_size,
                        stream);
                break;
        }

        // _linearB
        Linear<T>::backward(
                buff_C,                         // C0, input_numel, in
                context_layer,
                linearB_weight,
                buff_E,                         // input_numel, out
                grad_linearB_weight,
                handle,
                batch_seq,
                hidden_size,
                hidden_size);

        /******************************************************************************\
                                   BertSelfAttention backward
        \******************************************************************************/

        Permute<T>::backward(
                buff_E,                         // input_numel, in
                buff_D,                         // D1, input_numel, out
                batch_size,
                seq_len,
                hidden_size,
                num_attention_heads,
                attention_head_size,
                stream);

        if (isActive(attention_probs_dropout_prob)) {
            // _gemm_pv
            BatchGemm<T, false, false>::backward(
                    buff_D,                     // D1, input_numel, in
                    dropout_out,
                    v_layer,
                    buff_F,                     // probs_numel, out
                    buff_CV,                    // C1, input_numel, out
                    handle,
                    batch_size * num_attention_heads,
                    seq_len,
                    attention_head_size,
                    seq_len);

            Dropout<T>::backward(
                    buff_F,                     // probs_numel, inout
                    dropout_mask,
                    toScale(attention_probs_dropout_prob),
                    get_probs_numel(batch_size, seq_len, num_attention_heads),
                    stream);
        }
        else {
            // _gemm_pv
            BatchGemm<T, false, false>::backward(
                    buff_D,                     // D1, input_numel, in
                    softmax_out,
                    v_layer,
                    buff_F,                     // probs_numel, out
                    buff_CV,                    // C1, input_numel, out
                    handle,
                    batch_size * num_attention_heads,
                    seq_len,
                    attention_head_size,
                    seq_len);
        }

        Softmax<T>::backward(
                softmax_out,
                buff_F,                         // probs_numel, inout
                batch_size * num_attention_heads * seq_len,
                seq_len,
                stream);

        // _gemm_qk
        BatchGemm<T, false, true>::backward(
                buff_F,                         // probs_numel, in
                q_layer,
                k_layer,
                buff_CQ,                        // C1, input_numel, out
                buff_CK,                        // C1, input_numel, out
                handle,
                batch_size * num_attention_heads,
                seq_len,
                seq_len,
                attention_head_size,
                1. / sqrt(attention_head_size));

        BiasedPermute<T>::backward(
                buff_C,                         // C1, 3 * input_numel, in
                buff_G,                         // 3 * input_numel, out
                grad_linearA_bias,
                batch_size,
                seq_len,
                hidden_size,
                num_attention_heads,
                attention_head_size,
                stream);

        // _linearA
        Linear<T>::backwardA(
                buff_G,                         // 3 * input_numel, in
                hidden_states,
                linearA_weight,
                grad_hidden_states,
                grad_linearA_weight,
                handle,
                batch_seq,
                hidden_size,
                3 * hidden_size);
    }
public:
    static int64_t
    get_f_buff_size(int64_t batch_size,
                    int64_t seq_len,
                    int64_t hidden_size,
                    int64_t intermediate_size) {
        bool reusable = is_reusable(hidden_size, intermediate_size);
        int64_t input_numel = get_input_numel(batch_size, seq_len, hidden_size);
        int64_t f_buff_size = reusable ? input_numel * 2 : input_numel * 3;
        return f_buff_size;
    }

    static int64_t
    get_b_buff_size(int64_t batch_size,
                    int64_t seq_len,
                    int64_t hidden_size,
                    int64_t intermediate_size) {
        bool reusable = is_reusable(hidden_size, intermediate_size);
        int64_t b_buff_size = reusable ? 0 : get_input_numel(batch_size, seq_len, hidden_size) * 3;
        return b_buff_size;
    }

    template <typename T>
    static void
    forward(T* buffer,
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
            T* qkv_layer,
            T* softmax_out,
            T* dropout_out,                     // do_P_dropout, backbuff
            uint8_t* dropout_mask,              // do_P_dropout
            uint8_t* dropoutA_mask,             // do_H_dropout
            uint8_t* dropoutB_mask,             // do_H_dropout
            uint8_t* dropPathA_mask,            // do_drop_path
            uint8_t* dropPathB_mask,            // do_drop_path
            T* normA_out,
            T* normA_rstd,
            T* normB_out,                       // pre_layernorm, backbuff
            T* normB_rstd,
            T* context_layer,
            T* gelu_inp,
            T* gelu_out,
            T* layer_out,
            int64_t batch_size,
            int64_t seq_len,
            int64_t hidden_size,
            bool pre_layernorm,
            int64_t num_attention_heads,
            int64_t intermediate_size,
            double attention_probs_dropout_prob,
            double hidden_dropout_prob,
            double drop_path_prob,
            float layer_norm_eps,
            cublasHandle_t handle,
            cudaStream_t stream) {
        auto forward_func = pre_layernorm ? pre_layernorm_forward<T> : post_layernorm_forward<T>;
        forward_func(
                buffer,
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
                qkv_layer,
                softmax_out,
                dropout_out,
                dropout_mask,
                dropoutA_mask,
                dropoutB_mask,
                dropPathA_mask,
                dropPathB_mask,
                normA_out,
                normA_rstd,
                normB_out,
                normB_rstd,
                context_layer,
                gelu_inp,
                gelu_out,
                layer_out,
                batch_size,
                seq_len,
                hidden_size,
                num_attention_heads,
                intermediate_size,
                attention_probs_dropout_prob,
                hidden_dropout_prob,
                drop_path_prob,
                layer_norm_eps,
                handle,
                stream);
    }

    template <typename T>
    static void
    backward(T* buffer,
             const T* grad,
             const T* hidden_states,            // !pre_layernorm
             const T* normA_gamma,
             const T* normA_beta,
             const T* normB_gamma,
             const T* normB_beta,
             const T* linearA_weight,
             const T* linearB_weight,
             const T* linearC_weight,
             const T* linearC_bias,
             const T* linearD_weight,
             T* qkv_layer,                      // in, buff
             const T* softmax_out,
             T* dropout_out,                    // in, buff
             const uint8_t* dropout_mask,       // do_P_dropout
             const uint8_t* dropoutA_mask,      // do_H_dropout
             const uint8_t* dropoutB_mask,      // do_H_dropout
             const uint8_t* dropPathA_mask,     // do_drop_path
             const uint8_t* dropPathB_mask,     // do_drop_path
             const T* normA_out,
             const T* normA_rstd,
             T* normB_out,                      // in, buff
             const T* normB_rstd,
             T* context_layer,                  // in, buff
             T* gelu_inp,                       // in, buff
             T* gelu_out,                       // in, buff
             const T* layer_out,                // !pre_layernorm
             T* grad_hidden_states,             // out, buff
             T* grad_normA_gamma,
             T* grad_normA_beta,
             T* grad_normB_gamma,
             T* grad_normB_beta,
             T* grad_linearA_weight,
             T* grad_linearA_bias,
             T* grad_linearB_weight,
             T* grad_linearB_bias,
             T* grad_linearC_weight,
             T* grad_linearC_bias,
             T* grad_linearD_weight,
             T* grad_linearD_bias,
             int64_t batch_size,
             int64_t seq_len,
             int64_t hidden_size,
             bool pre_layernorm,
             int64_t num_attention_heads,
             int64_t intermediate_size,
             double attention_probs_dropout_prob,
             double hidden_dropout_prob,
             double drop_path_prob,
             cublasHandle_t handle,
             cudaStream_t stream) {
        auto backward_func = pre_layernorm ? pre_layernorm_backward<T> : post_layernorm_backward<T>;
        backward_func(
                buffer,
                grad,
                hidden_states,
                normA_gamma,
                normA_beta,
                normB_gamma,
                normB_beta,
                linearA_weight,
                linearB_weight,
                linearC_weight,
                linearC_bias,
                linearD_weight,
                qkv_layer,
                softmax_out,
                dropout_out,
                dropout_mask,
                dropoutA_mask,
                dropoutB_mask,
                dropPathA_mask,
                dropPathB_mask,
                normA_out,
                normA_rstd,
                normB_out,
                normB_rstd,
                context_layer,
                gelu_inp,
                gelu_out,
                layer_out,
                grad_hidden_states,
                grad_normA_gamma,
                grad_normA_beta,
                grad_normB_gamma,
                grad_normB_beta,
                grad_linearA_weight,
                grad_linearA_bias,
                grad_linearB_weight,
                grad_linearB_bias,
                grad_linearC_weight,
                grad_linearC_bias,
                grad_linearD_weight,
                grad_linearD_bias,
                batch_size,
                seq_len,
                hidden_size,
                num_attention_heads,
                intermediate_size,
                attention_probs_dropout_prob,
                hidden_dropout_prob,
                drop_path_prob,
                handle,
                stream);
    }
};

