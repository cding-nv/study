#include <torch/extension.h>
using torch::autograd::tensor_list;


void
MMSelfAttn_init(
        int64_t stream_cnt,
        bool fast_fp32);


torch::Tensor
MMSelfAttnInferL(
        const torch::Tensor& modal_index,
        const torch::Tensor& query_layer,
        const torch::Tensor& key_layer,
        const torch::Tensor& value_layer,
        const torch::Tensor& local_attention_mask,
        torch::Tensor& buff,
        bool use_multistream,
        bool fast_fp32);


torch::Tensor
MMSelfAttnTrainL(
        const torch::Tensor& modal_index,
        const torch::Tensor& query_layer,
        const torch::Tensor& key_layer,
        const torch::Tensor& value_layer,
        const torch::Tensor& local_attention_mask,
        double attention_probs_dropout_prob,
        bool use_multistream);


torch::Tensor
MMSelfAttnInferGL(
        const torch::Tensor& modal_index,
        const torch::Tensor& query_layer,
        const torch::Tensor& key_layer,
        const torch::Tensor& value_layer,
        const torch::Tensor& local_attention_mask,
        const torch::Tensor& global_k,
        const torch::Tensor& global_v,
        const torch::Tensor& global_selection_padding_mask_zeros,
        torch::Tensor& buff,
        bool use_multistream,
        bool fast_fp32);


torch::Tensor
MMSelfAttnTrainGL(
        const torch::Tensor& modal_index,
        const torch::Tensor& query_layer,
        const torch::Tensor& key_layer,
        const torch::Tensor& value_layer,
        const torch::Tensor& local_attention_mask,
        const torch::Tensor& global_k,
        const torch::Tensor& global_v,
        const torch::Tensor& global_selection_padding_mask_zeros,
        double attention_probs_dropout_prob,
        bool use_multistream);


TORCH_LIBRARY(mmsa, m) {
    m.def("MMSelfAttn_init", MMSelfAttn_init);
    m.def("MMSelfAttnInferL", MMSelfAttnInferL);
    m.def("MMSelfAttnTrainL", MMSelfAttnTrainL);
    m.def("MMSelfAttnInferGL", MMSelfAttnInferGL);
    m.def("MMSelfAttnTrainGL", MMSelfAttnTrainGL);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

