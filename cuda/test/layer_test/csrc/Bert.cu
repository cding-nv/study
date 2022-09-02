#include "BertEncoderInfer.h"
#include "BertEncoderTrain.h"
#include "BertPoolerInfer.h"
#include "BertPoolerTrain.h"
#include <ATen/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
using torch::autograd::tensor_list;


void
encoder_init(bool fast_fp32) {
    uint64_t seed = at::cuda::detail::getDefaultCUDAGenerator().current_seed();
    Context::instance().setSeed(seed);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    Context::instance().setHandle(stream, fast_fp32);
}


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
        int64_t num_hidden_layers) {
    CHECK_INPUT(hidden_states);
    CHECK_INPUT(attention_mask);
    CHECK_INPUT(normA_gamma_list);
    CHECK_INPUT(normA_beta_list);
    CHECK_INPUT(normB_gamma_list);
    CHECK_INPUT(normB_beta_list);
    CHECK_INPUT(linearA_weight_list);
    CHECK_INPUT(linearA_bias_list);
    CHECK_INPUT(linearB_weight_list);
    CHECK_INPUT(linearB_bias_list);
    CHECK_INPUT(linearC_weight_list);
    CHECK_INPUT(linearC_bias_list);
    CHECK_INPUT(linearD_weight_list);
    CHECK_INPUT(linearD_bias_list);
    if (pre_layernorm) {
        CHECK_INPUT(norm_gamma);
        CHECK_INPUT(norm_beta);
    }
    auto float_options = torch::TensorOptions().dtype(hidden_states.dtype()).device(hidden_states.device());

    const int64_t batch_size = hidden_states.size(0);
    const int64_t seq_len = hidden_states.size(1);
    const int64_t hidden_size = hidden_states.size(2);
    const int64_t buff_size = BertEncoderInfer::get_f_buff_size(batch_size, seq_len, hidden_size, num_attention_heads, intermediate_size);

    torch::Tensor buff = torch::empty(buff_size, float_options);
    torch::Tensor sequence_output = torch::empty({batch_size, seq_len, hidden_size}, float_options);
    torch::Tensor layer_out_list = torch::empty({num_hidden_layers, batch_size, seq_len, hidden_size}, float_options);

    if (hidden_states.scalar_type() == at::ScalarType::Half) {
        BertEncoderInfer::forward<__half>(
                at::cuda::getCurrentCUDAStream(),
                (const __half*)hidden_states.data_ptr(),
                (const __half*)attention_mask.data_ptr(),
                (const __half*)normA_gamma_list.data_ptr(),
                (const __half*)normA_beta_list.data_ptr(),
                (const __half*)normB_gamma_list.data_ptr(),
                (const __half*)normB_beta_list.data_ptr(),
                (const __half*)linearA_weight_list.data_ptr(),
                (const __half*)linearA_bias_list.data_ptr(),
                (const __half*)linearB_weight_list.data_ptr(),
                (const __half*)linearB_bias_list.data_ptr(),
                (const __half*)linearC_weight_list.data_ptr(),
                (const __half*)linearC_bias_list.data_ptr(),
                (const __half*)linearD_weight_list.data_ptr(),
                (const __half*)linearD_bias_list.data_ptr(),
                (const __half*)norm_gamma.data_ptr(),
                (const __half*)norm_beta.data_ptr(),
                (__half*)sequence_output.data_ptr(),
                (__half*)layer_out_list.data_ptr(),
                (__half*)buff.data_ptr(),
                batch_size,
                seq_len,
                hidden_size,
                pre_layernorm,
                num_attention_heads,
                intermediate_size,
                layer_norm_eps,
                fast_fp32,
                num_hidden_layers);
    }
    else if (hidden_states.scalar_type() == at::ScalarType::Float) {
        BertEncoderInfer::forward<float>(
                at::cuda::getCurrentCUDAStream(),
                (const float*)hidden_states.data_ptr(),
                (const float*)attention_mask.data_ptr(),
                (const float*)normA_gamma_list.data_ptr(),
                (const float*)normA_beta_list.data_ptr(),
                (const float*)normB_gamma_list.data_ptr(),
                (const float*)normB_beta_list.data_ptr(),
                (const float*)linearA_weight_list.data_ptr(),
                (const float*)linearA_bias_list.data_ptr(),
                (const float*)linearB_weight_list.data_ptr(),
                (const float*)linearB_bias_list.data_ptr(),
                (const float*)linearC_weight_list.data_ptr(),
                (const float*)linearC_bias_list.data_ptr(),
                (const float*)linearD_weight_list.data_ptr(),
                (const float*)linearD_bias_list.data_ptr(),
                (const float*)norm_gamma.data_ptr(),
                (const float*)norm_beta.data_ptr(),
                (float*)sequence_output.data_ptr(),
                (float*)layer_out_list.data_ptr(),
                (float*)buff.data_ptr(),
                batch_size,
                seq_len,
                hidden_size,
                pre_layernorm,
                num_attention_heads,
                intermediate_size,
                layer_norm_eps,
                fast_fp32,
                num_hidden_layers);
    }
    else {
        errMsg("invalid dtype");
    }

    return {sequence_output,
            layer_out_list};
}


class EncoderAutograd: public torch::autograd::Function<EncoderAutograd> {
public:
    static tensor_list
    forward(torch::autograd::AutogradContext* ctx,
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
            double layer_norm_eps) {
        CHECK_INPUT(hidden_states);
        CHECK_INPUT(attention_mask);
        CHECK_INPUT(normA_gamma_list);
        CHECK_INPUT(normA_beta_list);
        CHECK_INPUT(normB_gamma_list);
        CHECK_INPUT(normB_beta_list);
        CHECK_INPUT(linearA_weight_list);
        CHECK_INPUT(linearA_bias_list);
        CHECK_INPUT(linearB_weight_list);
        CHECK_INPUT(linearB_bias_list);
        CHECK_INPUT(linearC_weight_list);
        CHECK_INPUT(linearC_bias_list);
        CHECK_INPUT(linearD_weight_list);
        CHECK_INPUT(linearD_bias_list);
        if (pre_layernorm) {
            CHECK_INPUT(norm_gamma);
            CHECK_INPUT(norm_beta);
        }
        auto float_options = torch::TensorOptions().dtype(hidden_states.dtype()).device(hidden_states.device());
        auto uint8_options = torch::TensorOptions().dtype(torch::kUInt8).device(hidden_states.device()).requires_grad(false);
        torch::Tensor ghost = torch::empty(0, float_options);

        const int64_t batch_size = hidden_states.size(0);
        const int64_t seq_len = hidden_states.size(1);
        const int64_t hidden_size = hidden_states.size(2);
        const int64_t num_hidden_layers = normA_gamma_list.size(0);
        const bool do_P_dropout = isActive(attention_probs_dropout_prob);
        const bool do_H_dropout = isActive(hidden_dropout_prob);
        const bool do_drop_path = isActive(drop_path_prob);
        const int64_t batch_seq = batch_size * seq_len;
        const int64_t input_numel = BertCom::get_input_numel(batch_size, seq_len, hidden_size);
        const int64_t probs_numel = BertCom::get_probs_numel(batch_size, seq_len, num_attention_heads);
        const int64_t media_numel = BertCom::get_media_numel(batch_size, seq_len, intermediate_size);
        const int64_t buff_size = BertEncoderTrain::get_f_buff_size(batch_size, seq_len, hidden_size, intermediate_size);

        torch::Tensor buff = torch::empty(buff_size, float_options);
        torch::Tensor qkv_layer_list = torch::empty({num_hidden_layers, 3 * input_numel}, float_options);
        torch::Tensor softmax_out_list = torch::empty({num_hidden_layers, batch_size, num_attention_heads, seq_len, seq_len}, float_options);
        torch::Tensor dropout_out_list = do_P_dropout ? torch::empty({num_hidden_layers, batch_size, num_attention_heads, seq_len, seq_len}, float_options) : torch::empty(probs_numel, float_options);
        torch::Tensor dropout_mask_list = do_P_dropout ? torch::empty({num_hidden_layers, probs_numel}, uint8_options) : ghost;
        torch::Tensor dropoutA_mask_list = do_H_dropout ? torch::empty({num_hidden_layers, input_numel}, uint8_options) : ghost;
        torch::Tensor dropoutB_mask_list = do_H_dropout ? torch::empty({num_hidden_layers, input_numel}, uint8_options) : ghost;
        torch::Tensor dropPathA_mask_list = do_drop_path ? torch::empty({num_hidden_layers, batch_size}, uint8_options) : ghost;
        torch::Tensor dropPathB_mask_list = do_drop_path ? torch::empty({num_hidden_layers, batch_size}, uint8_options) : ghost;
        torch::Tensor normA_out_list = torch::empty({num_hidden_layers, input_numel}, float_options);
        torch::Tensor normA_rstd_list = torch::empty({num_hidden_layers, batch_seq}, float_options);
        torch::Tensor normB_out_list = pre_layernorm ? torch::empty({num_hidden_layers, input_numel}, float_options) : torch::empty(input_numel, float_options);
        torch::Tensor normB_rstd_list = torch::empty({num_hidden_layers, batch_seq}, float_options);
        torch::Tensor context_layer_list = torch::empty({num_hidden_layers, input_numel}, float_options);
        torch::Tensor gelu_inp_list = torch::empty({num_hidden_layers, media_numel}, float_options);
        torch::Tensor gelu_out_list = torch::empty({num_hidden_layers, media_numel}, float_options);
        torch::Tensor layer_out_list = torch::empty({num_hidden_layers, batch_size, seq_len, hidden_size}, float_options);
        torch::Tensor norm_out = pre_layernorm ? torch::empty({batch_size, seq_len, hidden_size}, float_options) : ghost;
        torch::Tensor norm_rstd = pre_layernorm ? torch::empty(batch_seq, float_options) : ghost;

        if (hidden_states.scalar_type() == at::ScalarType::Half) {
            BertEncoderTrain::forward<__half>(
                    (__half*)buff.data_ptr(),
                    (const __half*)hidden_states.data_ptr(),
                    (const __half*)attention_mask.data_ptr(),
                    (const __half*)normA_gamma_list.data_ptr(),
                    (const __half*)normA_beta_list.data_ptr(),
                    (const __half*)normB_gamma_list.data_ptr(),
                    (const __half*)normB_beta_list.data_ptr(),
                    (const __half*)linearA_weight_list.data_ptr(),
                    (const __half*)linearA_bias_list.data_ptr(),
                    (const __half*)linearB_weight_list.data_ptr(),
                    (const __half*)linearB_bias_list.data_ptr(),
                    (const __half*)linearC_weight_list.data_ptr(),
                    (const __half*)linearC_bias_list.data_ptr(),
                    (const __half*)linearD_weight_list.data_ptr(),
                    (const __half*)linearD_bias_list.data_ptr(),
                    (const __half*)norm_gamma.data_ptr(),
                    (const __half*)norm_beta.data_ptr(),
                    (__half*)qkv_layer_list.data_ptr(),
                    (__half*)softmax_out_list.data_ptr(),
                    (__half*)dropout_out_list.data_ptr(),
                    (uint8_t*)dropout_mask_list.data_ptr(),
                    (uint8_t*)dropoutA_mask_list.data_ptr(),
                    (uint8_t*)dropoutB_mask_list.data_ptr(),
                    (uint8_t*)dropPathA_mask_list.data_ptr(),
                    (uint8_t*)dropPathB_mask_list.data_ptr(),
                    (__half*)normA_out_list.data_ptr(),
                    (__half*)normA_rstd_list.data_ptr(),
                    (__half*)normB_out_list.data_ptr(),
                    (__half*)normB_rstd_list.data_ptr(),
                    (__half*)context_layer_list.data_ptr(),
                    (__half*)gelu_inp_list.data_ptr(),
                    (__half*)gelu_out_list.data_ptr(),
                    (__half*)layer_out_list.data_ptr(),
                    (__half*)norm_out.data_ptr(),
                    (__half*)norm_rstd.data_ptr(),
                    batch_size,
                    seq_len,
                    hidden_size,
                    pre_layernorm,
                    num_attention_heads,
                    intermediate_size,
                    attention_probs_dropout_prob,
                    hidden_dropout_prob,
                    drop_path_prob,
                    layer_norm_eps,
                    num_hidden_layers,
                    at::cuda::getCurrentCUDAStream());
        }
        else if (hidden_states.scalar_type() == at::ScalarType::Float) {
            BertEncoderTrain::forward<float>(
                    (float*)buff.data_ptr(),
                    (const float*)hidden_states.data_ptr(),
                    (const float*)attention_mask.data_ptr(),
                    (const float*)normA_gamma_list.data_ptr(),
                    (const float*)normA_beta_list.data_ptr(),
                    (const float*)normB_gamma_list.data_ptr(),
                    (const float*)normB_beta_list.data_ptr(),
                    (const float*)linearA_weight_list.data_ptr(),
                    (const float*)linearA_bias_list.data_ptr(),
                    (const float*)linearB_weight_list.data_ptr(),
                    (const float*)linearB_bias_list.data_ptr(),
                    (const float*)linearC_weight_list.data_ptr(),
                    (const float*)linearC_bias_list.data_ptr(),
                    (const float*)linearD_weight_list.data_ptr(),
                    (const float*)linearD_bias_list.data_ptr(),
                    (const float*)norm_gamma.data_ptr(),
                    (const float*)norm_beta.data_ptr(),
                    (float*)qkv_layer_list.data_ptr(),
                    (float*)softmax_out_list.data_ptr(),
                    (float*)dropout_out_list.data_ptr(),
                    (uint8_t*)dropout_mask_list.data_ptr(),
                    (uint8_t*)dropoutA_mask_list.data_ptr(),
                    (uint8_t*)dropoutB_mask_list.data_ptr(),
                    (uint8_t*)dropPathA_mask_list.data_ptr(),
                    (uint8_t*)dropPathB_mask_list.data_ptr(),
                    (float*)normA_out_list.data_ptr(),
                    (float*)normA_rstd_list.data_ptr(),
                    (float*)normB_out_list.data_ptr(),
                    (float*)normB_rstd_list.data_ptr(),
                    (float*)context_layer_list.data_ptr(),
                    (float*)gelu_inp_list.data_ptr(),
                    (float*)gelu_out_list.data_ptr(),
                    (float*)layer_out_list.data_ptr(),
                    (float*)norm_out.data_ptr(),
                    (float*)norm_rstd.data_ptr(),
                    batch_size,
                    seq_len,
                    hidden_size,
                    pre_layernorm,
                    num_attention_heads,
                    intermediate_size,
                    attention_probs_dropout_prob,
                    hidden_dropout_prob,
                    drop_path_prob,
                    layer_norm_eps,
                    num_hidden_layers,
                    at::cuda::getCurrentCUDAStream());
        }
        else {
            errMsg("invalid dtype");
        }

        ctx->saved_data["normA_gamma_list"] = normA_gamma_list;
        ctx->saved_data["normA_beta_list"] = normA_beta_list;
        ctx->saved_data["normB_gamma_list"] = normB_gamma_list;
        ctx->saved_data["normB_beta_list"] = normB_beta_list;
        ctx->saved_data["linearA_weight_list"] = linearA_weight_list;
        ctx->saved_data["linearB_weight_list"] = linearB_weight_list;
        ctx->saved_data["linearC_weight_list"] = linearC_weight_list;
        ctx->saved_data["linearC_bias_list"] = linearC_bias_list;
        ctx->saved_data["linearD_weight_list"] = linearD_weight_list;
        ctx->saved_data["norm_gamma"] = norm_gamma;
        ctx->saved_data["norm_beta"] = norm_beta;
        ctx->saved_data["qkv_layer_list"] = qkv_layer_list;
        ctx->saved_data["dropout_mask_list"] = dropout_mask_list;
        ctx->saved_data["dropoutA_mask_list"] = dropoutA_mask_list;
        ctx->saved_data["dropoutB_mask_list"] = dropoutB_mask_list;
        ctx->saved_data["dropPathA_mask_list"] = dropPathA_mask_list;
        ctx->saved_data["dropPathB_mask_list"] = dropPathB_mask_list;
        ctx->saved_data["normA_out_list"] = normA_out_list;
        ctx->saved_data["normA_rstd_list"] = normA_rstd_list;
        ctx->saved_data["normB_out_list"] = normB_out_list;
        ctx->saved_data["normB_rstd_list"] = normB_rstd_list;
        ctx->saved_data["context_layer_list"] = context_layer_list;
        ctx->saved_data["gelu_inp_list"] = gelu_inp_list;
        ctx->saved_data["gelu_out_list"] = gelu_out_list;
        ctx->saved_data["norm_rstd"] = norm_rstd;
        ctx->saved_data["pre_layernorm"] = pre_layernorm;
        ctx->saved_data["intermediate_size"] = intermediate_size;
        ctx->saved_data["attention_probs_dropout_prob"] = attention_probs_dropout_prob;
        ctx->saved_data["hidden_dropout_prob"] = hidden_dropout_prob;
        ctx->saved_data["drop_path_prob"] = drop_path_prob;
        if (pre_layernorm) {
            ctx->save_for_backward({
                    softmax_out_list,
                    dropout_out_list,
                    layer_out_list,
                    norm_out});
        }
        else {
            ctx->save_for_backward({
                    softmax_out_list,
                    dropout_out_list,
                    layer_out_list,
                    hidden_states});
        }

        // construct outputs
        torch::Tensor sequence_output = pre_layernorm ? norm_out : layer_out_list[num_hidden_layers - 1];
        torch::Tensor attention_list = do_P_dropout ? dropout_out_list : softmax_out_list;
        ctx->mark_non_differentiable({layer_out_list, attention_list});
        return {sequence_output,
                layer_out_list,
                attention_list};
    }

    static tensor_list
    backward(torch::autograd::AutogradContext* ctx,
             const tensor_list& grad_list) {
        auto grad = grad_list[0];
        CHECK_INPUT(grad);
        auto float_options = torch::TensorOptions().dtype(grad.dtype()).device(grad.device());
        torch::Tensor ghost = torch::empty(0, float_options);

        auto normA_gamma_list = ctx->saved_data["normA_gamma_list"].toTensor();
        auto normA_beta_list = ctx->saved_data["normA_beta_list"].toTensor();
        auto normB_gamma_list = ctx->saved_data["normB_gamma_list"].toTensor();
        auto normB_beta_list = ctx->saved_data["normB_beta_list"].toTensor();
        auto linearA_weight_list = ctx->saved_data["linearA_weight_list"].toTensor();
        auto linearB_weight_list = ctx->saved_data["linearB_weight_list"].toTensor();
        auto linearC_weight_list = ctx->saved_data["linearC_weight_list"].toTensor();
        auto linearC_bias_list = ctx->saved_data["linearC_bias_list"].toTensor();
        auto linearD_weight_list = ctx->saved_data["linearD_weight_list"].toTensor();
        auto norm_gamma = ctx->saved_data["norm_gamma"].toTensor();
        auto norm_beta = ctx->saved_data["norm_beta"].toTensor();
        auto qkv_layer_list = ctx->saved_data["qkv_layer_list"].toTensor();
        auto dropout_mask_list = ctx->saved_data["dropout_mask_list"].toTensor();
        auto dropoutA_mask_list = ctx->saved_data["dropoutA_mask_list"].toTensor();
        auto dropoutB_mask_list = ctx->saved_data["dropoutB_mask_list"].toTensor();
        auto dropPathA_mask_list = ctx->saved_data["dropPathA_mask_list"].toTensor();
        auto dropPathB_mask_list = ctx->saved_data["dropPathB_mask_list"].toTensor();
        auto normA_out_list = ctx->saved_data["normA_out_list"].toTensor();
        auto normA_rstd_list = ctx->saved_data["normA_rstd_list"].toTensor();
        auto normB_out_list = ctx->saved_data["normB_out_list"].toTensor();
        auto normB_rstd_list = ctx->saved_data["normB_rstd_list"].toTensor();
        auto context_layer_list = ctx->saved_data["context_layer_list"].toTensor();
        auto gelu_inp_list = ctx->saved_data["gelu_inp_list"].toTensor();
        auto gelu_out_list = ctx->saved_data["gelu_out_list"].toTensor();
        auto norm_rstd = ctx->saved_data["norm_rstd"].toTensor();
        auto pre_layernorm = ctx->saved_data["pre_layernorm"].toBool();
        auto intermediate_size = ctx->saved_data["intermediate_size"].toInt();
        auto attention_probs_dropout_prob = ctx->saved_data["attention_probs_dropout_prob"].toDouble();
        auto hidden_dropout_prob = ctx->saved_data["hidden_dropout_prob"].toDouble();
        auto drop_path_prob = ctx->saved_data["drop_path_prob"].toDouble();
        ctx->saved_data.clear();
        tensor_list saved_list = std::move(ctx->get_saved_variables());
        torch::Tensor& softmax_out_list = saved_list[0];
        torch::Tensor& dropout_out_list = saved_list[1];
        torch::Tensor& layer_out_list = saved_list[2];
        torch::Tensor& hidden_states = pre_layernorm ? ghost : saved_list[3];
        torch::Tensor& norm_out = pre_layernorm ? saved_list[3] : ghost;

        const int64_t batch_size = grad.size(0);
        const int64_t seq_len = grad.size(1);
        const int64_t hidden_size = grad.size(2);
        const int64_t num_attention_heads = softmax_out_list.size(2);
        const int64_t num_hidden_layers = softmax_out_list.size(0);
        const int64_t buff_size = BertEncoderTrain::get_b_buff_size(batch_size, seq_len, hidden_size, intermediate_size);

        torch::Tensor buff = torch::empty(buff_size, float_options);
        torch::Tensor grad_hidden_states = torch::empty({batch_size, seq_len, hidden_size}, float_options);
        torch::Tensor grad_normA_gamma_list = torch::empty({num_hidden_layers, hidden_size}, float_options);
        torch::Tensor grad_normA_beta_list = torch::empty({num_hidden_layers, hidden_size}, float_options);
        torch::Tensor grad_normB_gamma_list = torch::empty({num_hidden_layers, hidden_size}, float_options);
        torch::Tensor grad_normB_beta_list = torch::empty({num_hidden_layers, hidden_size}, float_options);
        torch::Tensor grad_linearA_weight_list = torch::empty({num_hidden_layers, 3 * hidden_size, hidden_size}, float_options);
        torch::Tensor grad_linearA_bias_list = torch::empty({num_hidden_layers, 3 * hidden_size}, float_options);
        torch::Tensor grad_linearB_weight_list = torch::empty({num_hidden_layers, hidden_size, hidden_size}, float_options);
        torch::Tensor grad_linearB_bias_list = torch::empty({num_hidden_layers, hidden_size}, float_options);
        torch::Tensor grad_linearC_weight_list = torch::empty({num_hidden_layers, intermediate_size, hidden_size}, float_options);
        torch::Tensor grad_linearC_bias_list = torch::empty({num_hidden_layers, intermediate_size}, float_options);
        torch::Tensor grad_linearD_weight_list = torch::empty({num_hidden_layers, hidden_size, intermediate_size}, float_options);
        torch::Tensor grad_linearD_bias_list = torch::empty({num_hidden_layers, hidden_size}, float_options);
        torch::Tensor grad_norm_gamma = pre_layernorm ? torch::empty(hidden_size, float_options) : ghost;
        torch::Tensor grad_norm_beta = pre_layernorm ? torch::empty(hidden_size, float_options) : ghost;

        if (grad.scalar_type() == at::ScalarType::Half) {
            BertEncoderTrain::backward<__half>(
                    (__half*)buff.data_ptr(),
                    (const __half*)grad.data_ptr(),
                    (const __half*)hidden_states.data_ptr(),
                    (const __half*)normA_gamma_list.data_ptr(),
                    (const __half*)normA_beta_list.data_ptr(),
                    (const __half*)normB_gamma_list.data_ptr(),
                    (const __half*)normB_beta_list.data_ptr(),
                    (const __half*)linearA_weight_list.data_ptr(),
                    (const __half*)linearB_weight_list.data_ptr(),
                    (const __half*)linearC_weight_list.data_ptr(),
                    (const __half*)linearC_bias_list.data_ptr(),
                    (const __half*)linearD_weight_list.data_ptr(),
                    (const __half*)norm_gamma.data_ptr(),
                    (const __half*)norm_beta.data_ptr(),
                    (__half*)qkv_layer_list.data_ptr(),
                    (const __half*)softmax_out_list.data_ptr(),
                    (__half*)dropout_out_list.data_ptr(),
                    (const uint8_t*)dropout_mask_list.data_ptr(),
                    (const uint8_t*)dropoutA_mask_list.data_ptr(),
                    (const uint8_t*)dropoutB_mask_list.data_ptr(),
                    (const uint8_t*)dropPathA_mask_list.data_ptr(),
                    (const uint8_t*)dropPathB_mask_list.data_ptr(),
                    (const __half*)normA_out_list.data_ptr(),
                    (const __half*)normA_rstd_list.data_ptr(),
                    (__half*)normB_out_list.data_ptr(),
                    (const __half*)normB_rstd_list.data_ptr(),
                    (__half*)context_layer_list.data_ptr(),
                    (__half*)gelu_inp_list.data_ptr(),
                    (__half*)gelu_out_list.data_ptr(),
                    (const __half*)layer_out_list.data_ptr(),
                    (const __half*)norm_out.data_ptr(),
                    (const __half*)norm_rstd.data_ptr(),
                    (__half*)grad_hidden_states.data_ptr(),
                    (__half*)grad_normA_gamma_list.data_ptr(),
                    (__half*)grad_normA_beta_list.data_ptr(),
                    (__half*)grad_normB_gamma_list.data_ptr(),
                    (__half*)grad_normB_beta_list.data_ptr(),
                    (__half*)grad_linearA_weight_list.data_ptr(),
                    (__half*)grad_linearA_bias_list.data_ptr(),
                    (__half*)grad_linearB_weight_list.data_ptr(),
                    (__half*)grad_linearB_bias_list.data_ptr(),
                    (__half*)grad_linearC_weight_list.data_ptr(),
                    (__half*)grad_linearC_bias_list.data_ptr(),
                    (__half*)grad_linearD_weight_list.data_ptr(),
                    (__half*)grad_linearD_bias_list.data_ptr(),
                    (__half*)grad_norm_gamma.data_ptr(),
                    (__half*)grad_norm_beta.data_ptr(),
                    batch_size,
                    seq_len,
                    hidden_size,
                    pre_layernorm,
                    num_attention_heads,
                    intermediate_size,
                    attention_probs_dropout_prob,
                    hidden_dropout_prob,
                    drop_path_prob,
                    num_hidden_layers,
                    at::cuda::getCurrentCUDAStream());
        }
        else if (grad.scalar_type() == at::ScalarType::Float) {
            BertEncoderTrain::backward<float>(
                    (float*)buff.data_ptr(),
                    (const float*)grad.data_ptr(),
                    (const float*)hidden_states.data_ptr(),
                    (const float*)normA_gamma_list.data_ptr(),
                    (const float*)normA_beta_list.data_ptr(),
                    (const float*)normB_gamma_list.data_ptr(),
                    (const float*)normB_beta_list.data_ptr(),
                    (const float*)linearA_weight_list.data_ptr(),
                    (const float*)linearB_weight_list.data_ptr(),
                    (const float*)linearC_weight_list.data_ptr(),
                    (const float*)linearC_bias_list.data_ptr(),
                    (const float*)linearD_weight_list.data_ptr(),
                    (const float*)norm_gamma.data_ptr(),
                    (const float*)norm_beta.data_ptr(),
                    (float*)qkv_layer_list.data_ptr(),
                    (const float*)softmax_out_list.data_ptr(),
                    (float*)dropout_out_list.data_ptr(),
                    (const uint8_t*)dropout_mask_list.data_ptr(),
                    (const uint8_t*)dropoutA_mask_list.data_ptr(),
                    (const uint8_t*)dropoutB_mask_list.data_ptr(),
                    (const uint8_t*)dropPathA_mask_list.data_ptr(),
                    (const uint8_t*)dropPathB_mask_list.data_ptr(),
                    (const float*)normA_out_list.data_ptr(),
                    (const float*)normA_rstd_list.data_ptr(),
                    (float*)normB_out_list.data_ptr(),
                    (const float*)normB_rstd_list.data_ptr(),
                    (float*)context_layer_list.data_ptr(),
                    (float*)gelu_inp_list.data_ptr(),
                    (float*)gelu_out_list.data_ptr(),
                    (const float*)layer_out_list.data_ptr(),
                    (const float*)norm_out.data_ptr(),
                    (const float*)norm_rstd.data_ptr(),
                    (float*)grad_hidden_states.data_ptr(),
                    (float*)grad_normA_gamma_list.data_ptr(),
                    (float*)grad_normA_beta_list.data_ptr(),
                    (float*)grad_normB_gamma_list.data_ptr(),
                    (float*)grad_normB_beta_list.data_ptr(),
                    (float*)grad_linearA_weight_list.data_ptr(),
                    (float*)grad_linearA_bias_list.data_ptr(),
                    (float*)grad_linearB_weight_list.data_ptr(),
                    (float*)grad_linearB_bias_list.data_ptr(),
                    (float*)grad_linearC_weight_list.data_ptr(),
                    (float*)grad_linearC_bias_list.data_ptr(),
                    (float*)grad_linearD_weight_list.data_ptr(),
                    (float*)grad_linearD_bias_list.data_ptr(),
                    (float*)grad_norm_gamma.data_ptr(),
                    (float*)grad_norm_beta.data_ptr(),
                    batch_size,
                    seq_len,
                    hidden_size,
                    pre_layernorm,
                    num_attention_heads,
                    intermediate_size,
                    attention_probs_dropout_prob,
                    hidden_dropout_prob,
                    drop_path_prob,
                    num_hidden_layers,
                    at::cuda::getCurrentCUDAStream());
        }
        else {
            errMsg("invalid dtype");
        }

        return {grad_hidden_states,
                torch::Tensor(),
                grad_normA_gamma_list,
                grad_normA_beta_list,
                grad_normB_gamma_list,
                grad_normB_beta_list,
                grad_linearA_weight_list,
                grad_linearA_bias_list,
                grad_linearB_weight_list,
                grad_linearB_bias_list,
                grad_linearC_weight_list,
                grad_linearC_bias_list,
                grad_linearD_weight_list,
                grad_linearD_bias_list,
                grad_norm_gamma,
                grad_norm_beta,
                torch::Tensor(),
                torch::Tensor(),
                torch::Tensor(),
                torch::Tensor(),
                torch::Tensor(),
                torch::Tensor(),
                torch::Tensor()};
    }
};


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
        double layer_norm_eps) {
    return EncoderAutograd::apply(
            hidden_states,
            attention_mask,
            normA_gamma_list,
            normA_beta_list,
            normB_gamma_list,
            normB_beta_list,
            linearA_weight_list,
            linearA_bias_list,
            linearB_weight_list,
            linearB_bias_list,
            linearC_weight_list,
            linearC_bias_list,
            linearD_weight_list,
            linearD_bias_list,
            norm_gamma,
            norm_beta,
            pre_layernorm,
            num_attention_heads,
            intermediate_size,
            attention_probs_dropout_prob,
            hidden_dropout_prob,
            drop_path_prob,
            layer_norm_eps);
}


void
pooler_init(bool fast_fp32) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    Context::instance().setHandle(stream, fast_fp32);
}


torch::Tensor
pooler_infer(
        const torch::Tensor& hidden_states,
        const torch::Tensor& linear_weight,
        const torch::Tensor& linear_bias,
        int64_t cls_count,
        bool fast_fp32) {
    CHECK_INPUT(hidden_states);
    CHECK_INPUT(linear_weight);
    CHECK_INPUT(linear_bias);
    auto float_options = torch::TensorOptions().dtype(hidden_states.dtype()).device(hidden_states.device());

    const int64_t batch_size = hidden_states.size(0);
    const int64_t seq_len = hidden_states.size(1);
    const int64_t hidden_size = hidden_states.size(2);
    const int64_t buff_size = BertPoolerInfer::get_f_buff_size(batch_size, hidden_size, cls_count);

    torch::Tensor buff = torch::empty(buff_size, float_options);
    torch::Tensor pooled_output = torch::empty({batch_size, cls_count, hidden_size}, float_options);

    if (hidden_states.scalar_type() == at::ScalarType::Half) {
        BertPoolerInfer::forward<__half>(
                at::cuda::getCurrentCUDAStream(),
                (const __half*)hidden_states.data_ptr(),
                (const __half*)linear_weight.data_ptr(),
                (const __half*)linear_bias.data_ptr(),
                (__half*)pooled_output.data_ptr(),
                (__half*)buff.data_ptr(),
                batch_size,
                seq_len,
                hidden_size,
                cls_count,
                fast_fp32);
    }
    else if (hidden_states.scalar_type() == at::ScalarType::Float) {
        BertPoolerInfer::forward<float>(
                at::cuda::getCurrentCUDAStream(),
                (const float*)hidden_states.data_ptr(),
                (const float*)linear_weight.data_ptr(),
                (const float*)linear_bias.data_ptr(),
                (float*)pooled_output.data_ptr(),
                (float*)buff.data_ptr(),
                batch_size,
                seq_len,
                hidden_size,
                cls_count,
                fast_fp32);
    }
    else {
        errMsg("invalid dtype");
    }

    return pooled_output;
}


class PoolerAutograd: public torch::autograd::Function<PoolerAutograd> {
public:
    static torch::Tensor
    forward(torch::autograd::AutogradContext* ctx,
            const torch::Tensor& hidden_states,
            const torch::Tensor& linear_weight,
            const torch::Tensor& linear_bias,
            int64_t cls_count) {
        CHECK_INPUT(hidden_states);
        CHECK_INPUT(linear_weight);
        CHECK_INPUT(linear_bias);
        auto float_options = torch::TensorOptions().dtype(hidden_states.dtype()).device(hidden_states.device());

        const int64_t batch_size = hidden_states.size(0);
        const int64_t seq_len = hidden_states.size(1);
        const int64_t hidden_size = hidden_states.size(2);
        const int64_t output_numel = BertCom::get_output_numel(batch_size, hidden_size, cls_count);

        torch::Tensor cls_token = torch::empty(output_numel, float_options);
        torch::Tensor pooled_output = torch::empty({batch_size, cls_count, hidden_size}, float_options);

        if (hidden_states.scalar_type() == at::ScalarType::Half) {
            BertPoolerTrain::forward<__half>(
                    (const __half*)hidden_states.data_ptr(),
                    (const __half*)linear_weight.data_ptr(),
                    (const __half*)linear_bias.data_ptr(),
                    (__half*)cls_token.data_ptr(),
                    (__half*)pooled_output.data_ptr(),
                    batch_size,
                    seq_len,
                    hidden_size,
                    cls_count,
                    at::cuda::getCurrentCUDAStream());
        }
        else if (hidden_states.scalar_type() == at::ScalarType::Float) {
            BertPoolerTrain::forward<float>(
                    (const float*)hidden_states.data_ptr(),
                    (const float*)linear_weight.data_ptr(),
                    (const float*)linear_bias.data_ptr(),
                    (float*)cls_token.data_ptr(),
                    (float*)pooled_output.data_ptr(),
                    batch_size,
                    seq_len,
                    hidden_size,
                    cls_count,
                    at::cuda::getCurrentCUDAStream());
        }
        else {
            errMsg("invalid dtype");
        }

        ctx->saved_data["linear_weight"] = linear_weight;
        ctx->saved_data["cls_token"] = cls_token;
        ctx->saved_data["seq_len"] = seq_len;
        ctx->save_for_backward({pooled_output});
        return pooled_output;
    }

    static tensor_list
    backward(torch::autograd::AutogradContext* ctx,
             const tensor_list& grad_list) {
        auto grad = grad_list[0];
        CHECK_INPUT(grad);
        auto float_options = torch::TensorOptions().dtype(grad.dtype()).device(grad.device());

        auto linear_weight = ctx->saved_data["linear_weight"].toTensor();
        auto cls_token = ctx->saved_data["cls_token"].toTensor();
        auto seq_len = ctx->saved_data["seq_len"].toInt();
        ctx->saved_data.clear();
        tensor_list saved_list = std::move(ctx->get_saved_variables());
        torch::Tensor& pooled_output = saved_list[0];

        const int64_t batch_size = grad.size(0);
        const int64_t hidden_size = grad.size(-1);
        const int64_t cls_count = grad.dim() == 2 ? 1 : grad.size(1);

        torch::Tensor grad_hidden_states = torch::empty({batch_size, seq_len, hidden_size}, float_options);
        torch::Tensor grad_linear_weight = torch::empty({hidden_size, hidden_size}, float_options);
        torch::Tensor grad_linear_bias = torch::empty(hidden_size, float_options);

        if (grad.scalar_type() == at::ScalarType::Half) {
            BertPoolerTrain::backward<__half>(
                    (const __half*)grad.data_ptr(),
                    (const __half*)linear_weight.data_ptr(),
                    (__half*)cls_token.data_ptr(),
                    (const __half*)pooled_output.data_ptr(),
                    (__half*)grad_hidden_states.data_ptr(),
                    (__half*)grad_linear_weight.data_ptr(),
                    (__half*)grad_linear_bias.data_ptr(),
                    batch_size,
                    seq_len,
                    hidden_size,
                    cls_count,
                    at::cuda::getCurrentCUDAStream());
        }
        else if (grad.scalar_type() == at::ScalarType::Float) {
            BertPoolerTrain::backward<float>(
                    (const float*)grad.data_ptr(),
                    (const float*)linear_weight.data_ptr(),
                    (float*)cls_token.data_ptr(),
                    (const float*)pooled_output.data_ptr(),
                    (float*)grad_hidden_states.data_ptr(),
                    (float*)grad_linear_weight.data_ptr(),
                    (float*)grad_linear_bias.data_ptr(),
                    batch_size,
                    seq_len,
                    hidden_size,
                    cls_count,
                    at::cuda::getCurrentCUDAStream());
        }
        else {
            errMsg("invalid dtype");
        }

        return {grad_hidden_states,
                grad_linear_weight,
                grad_linear_bias,
                torch::Tensor()};
    }
};


torch::Tensor
pooler_train(
        const torch::Tensor& hidden_states,
        const torch::Tensor& linear_weight,
        const torch::Tensor& linear_bias,
        int64_t cls_count) {
    return PoolerAutograd::apply(
            hidden_states,
            linear_weight,
            linear_bias,
            cls_count);
}

