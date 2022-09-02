#pragma once
#include <cuda_runtime.h>


class BertEncoderInfer {
public:
    static int
    get_f_buff_size(int batch_size,
                    int seq_len,
                    int hidden_size,
                    int num_attention_heads,
                    int intermediate_size);

    template <typename T>
    static void
    forward(cudaStream_t stream,
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
            const T* norm_gamma,
            const T* norm_beta,
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
            int num_hidden_layers);
};

