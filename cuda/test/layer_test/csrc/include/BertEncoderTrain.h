#pragma once
#include "BertLayerTrain.h"


class BertEncoderTrain: public BertCom {
public:
    static int64_t
    get_f_buff_size(int64_t batch_size,
                    int64_t seq_len,
                    int64_t hidden_size,
                    int64_t intermediate_size) {
        return BertLayerTrain::get_f_buff_size(batch_size, seq_len, hidden_size, intermediate_size);
    }

    static int64_t
    get_b_buff_size(int64_t batch_size,
                    int64_t seq_len,
                    int64_t hidden_size,
                    int64_t intermediate_size) {
        bool reusable = is_reusable(hidden_size, intermediate_size);
        int64_t input_numel = get_input_numel(batch_size, seq_len, hidden_size);
        int64_t b_buff_size = reusable ? input_numel : input_numel << 2;
        return b_buff_size;
    }

    template <typename T>
    static void
    forward(T* buffer,
            const T* hidden_states,
            const T* attention_mask,
            const T* normA_gamma_list,
            const T* normA_beta_list,
            const T* normB_gamma_list,
            const T* normB_beta_list,
            const T* linearA_weight_list,
            const T* linearA_bias_list,
            const T* linearB_weight_list,
            const T* linearB_bias_list,
            const T* linearC_weight_list,
            const T* linearC_bias_list,
            const T* linearD_weight_list,
            const T* linearD_bias_list,
            const T* norm_gamma,                // pre_layernorm
            const T* norm_beta,                 // pre_layernorm
            T* qkv_layer_list,
            T* softmax_out_list,
            T* dropout_out_list,                // do_P_dropout, backbuff
            uint8_t* dropout_mask_list,         // do_P_dropout
            uint8_t* dropoutA_mask_list,        // do_H_dropout
            uint8_t* dropoutB_mask_list,        // do_H_dropout
            uint8_t* dropPathA_mask_list,       // do_drop_path
            uint8_t* dropPathB_mask_list,       // do_drop_path
            T* normA_out_list,
            T* normA_rstd_list,
            T* normB_out_list,                  // pre_layernorm, backbuff
            T* normB_rstd_list,
            T* context_layer_list,
            T* gelu_inp_list,
            T* gelu_out_list,
            T* layer_out_list,
            T* norm_out,                        // pre_layernorm
            T* norm_rstd,                       // pre_layernorm
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
            int64_t num_hidden_layers,
            cudaStream_t stream) {
        int64_t batch_seq = batch_size * seq_len;
        int64_t input_numel = get_input_numel(batch_size, seq_len, hidden_size);
        int64_t probs_numel = get_probs_numel(batch_size, seq_len, num_attention_heads);
        int64_t media_numel = get_media_numel(batch_size, seq_len, intermediate_size);
        bool do_P_dropout = isActive(attention_probs_dropout_prob);
        bool do_H_dropout = isActive(hidden_dropout_prob);
        bool do_drop_path = isActive(drop_path_prob);
        double drop_path_step = double(drop_path_prob) / (num_hidden_layers - 1);
        cublasHandle_t handle = Context::instance().getHandle(stream);

        const T* input = hidden_states;
        T* layer_out = layer_out_list;

        for (int64_t i = 0LL; i < num_hidden_layers; ++i) {
            BertLayerTrain::forward(
                    buffer,
                    input,
                    attention_mask,
                    normA_gamma_list    + i * hidden_size,
                    normA_beta_list     + i * hidden_size,
                    normB_gamma_list    + i * hidden_size,
                    normB_beta_list     + i * hidden_size,
                    linearA_weight_list + i * 3 * hidden_size * hidden_size,
                    linearA_bias_list   + i * 3 * hidden_size,
                    linearB_weight_list + i * hidden_size * hidden_size,
                    linearB_bias_list   + i * hidden_size,
                    linearC_weight_list + i * intermediate_size * hidden_size,
                    linearC_bias_list   + i * intermediate_size,
                    linearD_weight_list + i * hidden_size * intermediate_size,
                    linearD_bias_list   + i * hidden_size,
                    qkv_layer_list      + i * 3 * input_numel,
                    softmax_out_list    + i * probs_numel,
                    do_P_dropout  ? dropout_out_list    + i * probs_numel : nullptr,
                    do_P_dropout  ? dropout_mask_list   + i * probs_numel : nullptr,
                    do_H_dropout  ? dropoutA_mask_list  + i * input_numel : nullptr,
                    do_H_dropout  ? dropoutB_mask_list  + i * input_numel : nullptr,
                    do_drop_path  ? dropPathA_mask_list + i * batch_size  : nullptr,
                    do_drop_path  ? dropPathB_mask_list + i * batch_size  : nullptr,
                    normA_out_list      + i * input_numel,
                    normA_rstd_list     + i * batch_seq,
                    pre_layernorm ? normB_out_list      + i * input_numel : nullptr,
                    normB_rstd_list     + i * batch_seq,
                    context_layer_list  + i * input_numel,
                    gelu_inp_list       + i * media_numel,
                    gelu_out_list       + i * media_numel,
                    layer_out,
                    batch_size,
                    seq_len,
                    hidden_size,
                    pre_layernorm,
                    num_attention_heads,
                    intermediate_size,
                    attention_probs_dropout_prob,
                    hidden_dropout_prob,
                    i * drop_path_step,
                    layer_norm_eps,
                    handle,
                    stream);
            input = layer_out;
            layer_out += input_numel;
        }

        if (pre_layernorm) {
            LayerNorm<T>::forward(
                    input,
                    norm_gamma,
                    norm_beta,
                    norm_out,
                    norm_rstd,
                    true,
                    layer_norm_eps,
                    batch_seq,
                    hidden_size,
                    stream);
        }
    }

    template <typename T>
    static void
    backward(T* buffer,
             const T* grad,
             const T* hidden_states,                    // !pre_layernorm
             const T* normA_gamma_list,
             const T* normA_beta_list,
             const T* normB_gamma_list,
             const T* normB_beta_list,
             const T* linearA_weight_list,
             const T* linearB_weight_list,
             const T* linearC_weight_list,
             const T* linearC_bias_list,
             const T* linearD_weight_list,
             const T* norm_gamma,                       // pre_layernorm
             const T* norm_beta,                        // pre_layernorm
             T* qkv_layer_list,                         // in, buff
             const T* softmax_out_list,
             T* dropout_out_list,                       // in, buff
             const uint8_t* dropout_mask_list,          // do_P_dropout
             const uint8_t* dropoutA_mask_list,         // do_H_dropout
             const uint8_t* dropoutB_mask_list,         // do_H_dropout
             const uint8_t* dropPathA_mask_list,        // do_drop_path
             const uint8_t* dropPathB_mask_list,        // do_drop_path
             const T* normA_out_list,
             const T* normA_rstd_list,
             T* normB_out_list,                         // in, buff
             const T* normB_rstd_list,
             T* context_layer_list,                     // in, buff
             T* gelu_inp_list,                          // in, buff
             T* gelu_out_list,                          // in, buff
             const T* layer_out_list,                   // !pre_layernorm
             const T* norm_out,                         // pre_layernorm
             const T* norm_rstd,                        // pre_layernorm
             T* grad_hidden_states,
             T* grad_normA_gamma_list,
             T* grad_normA_beta_list,
             T* grad_normB_gamma_list,
             T* grad_normB_beta_list,
             T* grad_linearA_weight_list,
             T* grad_linearA_bias_list,
             T* grad_linearB_weight_list,
             T* grad_linearB_bias_list,
             T* grad_linearC_weight_list,
             T* grad_linearC_bias_list,
             T* grad_linearD_weight_list,
             T* grad_linearD_bias_list,
             T* grad_norm_gamma,                        // pre_layernorm
             T* grad_norm_beta,                         // pre_layernorm
             int64_t batch_size,
             int64_t seq_len,
             int64_t hidden_size,
             bool pre_layernorm,
             int64_t num_attention_heads,
             int64_t intermediate_size,
             double attention_probs_dropout_prob,
             double hidden_dropout_prob,
             double drop_path_prob,
             int64_t num_hidden_layers,
             cudaStream_t stream) {
        int64_t batch_seq = batch_size * seq_len;
        int64_t input_numel = get_input_numel(batch_size, seq_len, hidden_size);
        int64_t probs_numel = get_probs_numel(batch_size, seq_len, num_attention_heads);
        int64_t media_numel = get_media_numel(batch_size, seq_len, intermediate_size);
        bool do_P_dropout = isActive(attention_probs_dropout_prob);
        bool do_H_dropout = isActive(hidden_dropout_prob);
        bool do_drop_path = isActive(drop_path_prob);
        double drop_path_step = double(drop_path_prob) / (num_hidden_layers - 1);
        cublasHandle_t handle = Context::instance().getHandle(stream);

        T *grad_inp, *grad_out;
        if (num_hidden_layers & 1) {
            grad_inp = buffer;
            grad_out = grad_hidden_states;
        }
        else {
            grad_inp = grad_hidden_states;
            grad_out = buffer;
        }

        if (pre_layernorm) {
            LayerNorm<T>::backward(
                    grad,
                    norm_out,
                    norm_gamma,
                    norm_beta,
                    norm_rstd,
                    grad_norm_gamma,
                    grad_norm_beta,
                    grad_inp,
                    batch_seq,
                    hidden_size,
                    stream);
            grad = grad_inp;
        }

        T* layer_buffer = buffer + input_numel;
        for (int64_t i = num_hidden_layers - 1; i >= 0; --i) {
            const T* input = pre_layernorm ? nullptr :
                             i ? layer_out_list + (i - 1) * input_numel :
                             hidden_states;
            BertLayerTrain::backward(
                    layer_buffer,
                    grad,
                    input,
                    normA_gamma_list         + i * hidden_size,
                    normA_beta_list          + i * hidden_size,
                    normB_gamma_list         + i * hidden_size,
                    normB_beta_list          + i * hidden_size,
                    linearA_weight_list      + i * 3 * hidden_size * hidden_size,
                    linearB_weight_list      + i * hidden_size * hidden_size,
                    linearC_weight_list      + i * intermediate_size * hidden_size,
                    linearC_bias_list        + i * intermediate_size,
                    linearD_weight_list      + i * hidden_size * intermediate_size,
                    qkv_layer_list           + i * 3 * input_numel,
                    softmax_out_list         + i * probs_numel,
                    do_P_dropout  ? dropout_out_list    + i * probs_numel : dropout_out_list,
                    do_P_dropout  ? dropout_mask_list   + i * probs_numel : nullptr,
                    do_H_dropout  ? dropoutA_mask_list  + i * input_numel : nullptr,
                    do_H_dropout  ? dropoutB_mask_list  + i * input_numel : nullptr,
                    do_drop_path  ? dropPathA_mask_list + i * batch_size  : nullptr,
                    do_drop_path  ? dropPathB_mask_list + i * batch_size  : nullptr,
                    normA_out_list           + i * input_numel,
                    normA_rstd_list          + i * batch_seq,
                    pre_layernorm ? normB_out_list      + i * input_numel : normB_out_list,
                    normB_rstd_list          + i * batch_seq,
                    context_layer_list       + i * input_numel,
                    gelu_inp_list            + i * media_numel,
                    gelu_out_list            + i * media_numel,
                    pre_layernorm ? nullptr : layer_out_list + i * input_numel,
                    grad_out,
                    grad_normA_gamma_list    + i * hidden_size,
                    grad_normA_beta_list     + i * hidden_size,
                    grad_normB_gamma_list    + i * hidden_size,
                    grad_normB_beta_list     + i * hidden_size,
                    grad_linearA_weight_list + i * 3 * hidden_size * hidden_size,
                    grad_linearA_bias_list   + i * 3 * hidden_size,
                    grad_linearB_weight_list + i * hidden_size * hidden_size,
                    grad_linearB_bias_list   + i * hidden_size,
                    grad_linearC_weight_list + i * intermediate_size * hidden_size,
                    grad_linearC_bias_list   + i * intermediate_size,
                    grad_linearD_weight_list + i * hidden_size * intermediate_size,
                    grad_linearD_bias_list   + i * hidden_size,
                    batch_size,
                    seq_len,
                    hidden_size,
                    pre_layernorm,
                    num_attention_heads,
                    intermediate_size,
                    attention_probs_dropout_prob,
                    hidden_dropout_prob,
                    i * drop_path_step,
                    handle,
                    stream);
            grad = grad_out;
            std::swap(grad_inp, grad_out);
        }
    }
};

