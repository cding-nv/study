#include <torch/extension.h>
using torch::autograd::tensor_list;


void
encoder_init(bool fast_fp32);


tensor_list
encoder_infer(
        const torch::Tensor& hidden_states,
        const torch::Tensor& attention_mask,
        const torch::Tensor& normA_gamma_list,
        const torch::Tensor& normA_beta_list,
        const torch::Tensor& normB_gamma_list,
        const torch::Tensor& normB_beta_list,
        const torch::Tensor& linearA_weight_list,
        const torch::Tensor& linearA_bias_list,
        const torch::Tensor& linearB_weight_list,
        const torch::Tensor& linearB_bias_list,
        const torch::Tensor& linearC_weight_list,
        const torch::Tensor& linearC_bias_list,
        const torch::Tensor& linearD_weight_list,
        const torch::Tensor& linearD_bias_list,
        const torch::Tensor& norm_gamma,
        const torch::Tensor& norm_beta,
        bool pre_layernorm,
        int64_t num_attention_heads,
        int64_t intermediate_size,
        double layer_norm_eps,
        bool fast_fp32,
        int64_t num_hidden_layers);


tensor_list
encoder_train(
        const torch::Tensor& hidden_states,
        const torch::Tensor& attention_mask,
        const torch::Tensor& normA_gamma_list,
        const torch::Tensor& normA_beta_list,
        const torch::Tensor& normB_gamma_list,
        const torch::Tensor& normB_beta_list,
        const torch::Tensor& linearA_weight_list,
        const torch::Tensor& linearA_bias_list,
        const torch::Tensor& linearB_weight_list,
        const torch::Tensor& linearB_bias_list,
        const torch::Tensor& linearC_weight_list,
        const torch::Tensor& linearC_bias_list,
        const torch::Tensor& linearD_weight_list,
        const torch::Tensor& linearD_bias_list,
        const torch::Tensor& norm_gamma,
        const torch::Tensor& norm_beta,
        bool pre_layernorm,
        int64_t num_attention_heads,
        int64_t intermediate_size,
        double attention_probs_dropout_prob,
        double hidden_dropout_prob,
        double drop_path_prob,
        double layer_norm_eps);


// for training
void
pooler_init(bool fast_fp32);


torch::Tensor
pooler_infer(
        const torch::Tensor& hidden_states,
        const torch::Tensor& linear_weight,
        const torch::Tensor& linear_bias,
        int64_t cls_count,
        bool fast_fp32);


torch::Tensor
pooler_train(
        const torch::Tensor& hidden_states,
        const torch::Tensor& linear_weight,
        const torch::Tensor& linear_bias,
        int64_t cls_count);


TORCH_LIBRARY(bert, m) {
    m.def("encoder_init", encoder_init);
    m.def("encoder_infer", encoder_infer);
    m.def("encoder_train", encoder_train);
    m.def("pooler_init", pooler_init);
    m.def("pooler_infer", pooler_infer);
    m.def("pooler_train", pooler_train);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

