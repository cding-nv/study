#include "BertEncoderInfer.h"
#include "BertLayerInfer.h"
#include "Context.h"


int
BertEncoderInfer::get_f_buff_size(
        int batch_size,
        int seq_len,
        int hidden_size,
        int num_attention_heads,
        int intermediate_size) {
    return BertLayerInfer::get_f_buff_size(batch_size, seq_len, hidden_size, num_attention_heads, intermediate_size);
}


template <typename T>
void
BertEncoderInfer::forward(
        cudaStream_t stream,
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
        T* sequence_output,
        T* layer_out_list,
        T* buffer,
        int batch_size,
        int seq_len,
        int hidden_size,
        bool pre_layernorm,
        int num_attention_heads,
        int intermediate_size,
        float layer_norm_eps,
        bool fast_fp32,
        int num_hidden_layers) {
    int input_numel = BertCom::get_input_numel(batch_size, seq_len, hidden_size);
    cublasHandle_t handle = Context::instance().getHandle(stream, fast_fp32);

    const T* input = hidden_states;
    T* layer_out = layer_out_list;

    for (int i = 0; i < num_hidden_layers; ++i) {
        BertLayerInfer::forward(
                stream,
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
                layer_out,
                buffer,
                batch_size,
                seq_len,
                hidden_size,
                pre_layernorm,
                num_attention_heads,
                intermediate_size,
                layer_norm_eps,
                handle);
        input = layer_out;
        layer_out += input_numel;
    }

    if (pre_layernorm) {
        LayerNorm<T>::forward(
                input,
                norm_gamma,
                norm_beta,
                sequence_output,
                nullptr,                    // norm_rstd
                false,                      // training
                layer_norm_eps,
                batch_size * seq_len,
                hidden_size,
                stream);
    }
    else {
        cudaMemcpyAsync(
                sequence_output,
                layer_out_list + (num_hidden_layers - 1) * input_numel,
                sizeof(T) * input_numel,
                cudaMemcpyDeviceToDevice,
                stream);
    }
}


template
void
BertEncoderInfer::forward(
        cudaStream_t stream,
        const float* hidden_states,
        const float* attention_mask,
        const float* normA_gamma_list,
        const float* normA_beta_list,
        const float* normB_gamma_list,
        const float* normB_beta_list,
        const float* linearA_weight_list,
        const float* linearA_bias_list,
        const float* linearB_weight_list,
        const float* linearB_bias_list,
        const float* linearC_weight_list,
        const float* linearC_bias_list,
        const float* linearD_weight_list,
        const float* linearD_bias_list,
        const float* norm_gamma,
        const float* norm_beta,
        float* sequence_output,
        float* layer_out_list,
        float* buffer,
        int batch_size,
        int seq_len,
        int hidden_size,
        bool pre_layernorm,
        int num_attention_heads,
        int intermediate_size,
        float layer_norm_eps,
        bool fast_fp32,
        int num_hidden_layers);


template
void
BertEncoderInfer::forward(
        cudaStream_t stream,
        const __half* hidden_states,
        const __half* attention_mask,
        const __half* normA_gamma_list,
        const __half* normA_beta_list,
        const __half* normB_gamma_list,
        const __half* normB_beta_list,
        const __half* linearA_weight_list,
        const __half* linearA_bias_list,
        const __half* linearB_weight_list,
        const __half* linearB_bias_list,
        const __half* linearC_weight_list,
        const __half* linearC_bias_list,
        const __half* linearD_weight_list,
        const __half* linearD_bias_list,
        const __half* norm_gamma,
        const __half* norm_beta,
        __half* sequence_output,
        __half* layer_out_list,
        __half* buffer,
        int batch_size,
        int seq_len,
        int hidden_size,
        bool pre_layernorm,
        int num_attention_heads,
        int intermediate_size,
        float layer_norm_eps,
        bool fast_fp32,
        int num_hidden_layers);

