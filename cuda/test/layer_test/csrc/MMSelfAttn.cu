#include "MMSelfAttnInferGL.h"
#include "MMSelfAttnInferL.h"
#include "MMSelfAttnTrainGL.h"
#include "MMSelfAttnTrainL.h"
#include <ATen/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
using torch::autograd::tensor_list;


void
MMSelfAttn_init(
        int64_t stream_cnt,
        bool fast_fp32) {
    uint64_t seed = at::cuda::detail::getDefaultCUDAGenerator().current_seed();
    Context::instance().setSeed(seed);
    Context::instance().setStreams(stream_cnt);
    for (int64_t i = 0LL; i < stream_cnt; ++i) {
        cudaStream_t stream = Context::instance().getStream(i);
        Context::instance().setHandle(stream, fast_fp32);
    }
}


torch::Tensor
MMSelfAttnInferL(
        const torch::Tensor& modal_index,
        const torch::Tensor& query_layer,
        const torch::Tensor& key_layer,
        const torch::Tensor& value_layer,
        const torch::Tensor& local_attention_mask,
        torch::Tensor& buff,
        bool use_multistream,
        bool fast_fp32) {
    CHECK_CONTIGUOUS(modal_index);      // host
    CHECK_INPUT(query_layer);
    CHECK_INPUT(key_layer);
    CHECK_INPUT(value_layer);
    CHECK_INPUT(local_attention_mask);
    auto float_options = torch::TensorOptions().dtype(query_layer.dtype()).device(query_layer.device());

    const int64_t modal_cnt = modal_index.size(0);
    const int64_t batch_size = query_layer.size(0);
    const int64_t seq_len = query_layer.size(1);
    const int64_t num_attention_heads = query_layer.size(2);
    const int64_t attention_head_size = query_layer.size(3);
    torch::Tensor context_layer = torch::zeros({batch_size, seq_len, num_attention_heads * attention_head_size}, float_options);

    if (query_layer.scalar_type() == at::ScalarType::Half) {
        MMSelfAttnInferL::forward<__half>(
                at::cuda::getCurrentCUDAStream(),
                (const int64_t*)modal_index.data_ptr(),
                (const __half*)query_layer.data_ptr(),
                (const __half*)key_layer.data_ptr(),
                (const __half*)value_layer.data_ptr(),
                (const __half*)local_attention_mask.data_ptr(),
                (__half*)context_layer.data_ptr(),
                (__half*)buff.data_ptr(),
                modal_cnt,
                batch_size,
                seq_len,
                num_attention_heads,
                attention_head_size,
                use_multistream,
                fast_fp32);
    }
    else if (query_layer.scalar_type() == at::ScalarType::Float) {
        MMSelfAttnInferL::forward<float>(
                at::cuda::getCurrentCUDAStream(),
                (const int64_t*)modal_index.data_ptr(),
                (const float*)query_layer.data_ptr(),
                (const float*)key_layer.data_ptr(),
                (const float*)value_layer.data_ptr(),
                (const float*)local_attention_mask.data_ptr(),
                (float*)context_layer.data_ptr(),
                (float*)buff.data_ptr(),
                modal_cnt,
                batch_size,
                seq_len,
                num_attention_heads,
                attention_head_size,
                use_multistream,
                fast_fp32);
    }
    else {
        errMsg("invalid dtype");
    }

    return context_layer;
}


class MMSelfAttnAutogradL: public torch::autograd::Function<MMSelfAttnAutogradL> {
public:
    static torch::Tensor
    forward(torch::autograd::AutogradContext* ctx,
            const torch::Tensor& modal_index,
            const torch::Tensor& query_layer,
            const torch::Tensor& key_layer,
            const torch::Tensor& value_layer,
            const torch::Tensor& local_attention_mask,
            double attention_probs_dropout_prob,
            bool use_multistream) {
        CHECK_CONTIGUOUS(modal_index);      // host
        CHECK_INPUT(query_layer);
        CHECK_INPUT(key_layer);
        CHECK_INPUT(value_layer);
        CHECK_INPUT(local_attention_mask);
        auto float_options = torch::TensorOptions().dtype(query_layer.dtype()).device(query_layer.device());
        auto uint8_options = torch::TensorOptions().dtype(torch::kUInt8).device(query_layer.device()).requires_grad(false);

        bool use_fp16;
        if (query_layer.scalar_type() == at::ScalarType::Half) {
            use_fp16 = true;
        }
        else if (query_layer.scalar_type() == at::ScalarType::Float) {
            use_fp16 = false;
        }
        else {
            errMsg("invalid dtype");
        }

        const int64_t batch_size = query_layer.size(0);
        const int64_t seq_len = query_layer.size(1);
        const int64_t num_attention_heads = query_layer.size(2);
        const int64_t attention_head_size = query_layer.size(3);
        const bool do_P_dropout = isActive(attention_probs_dropout_prob);

        MMSelfAttnCom com((const int64_t*)modal_index.data_ptr(),
                          modal_index.size(0),
                          batch_size,
                          num_attention_heads,
                          attention_head_size);
        const int64_t buff_size = MMSelfAttnTrainL::get_f_buff_size(use_multistream, com);
        const int64_t local_qkv_numel = com.get_local_qkv_numel();
        const int64_t local_attn_numel_align = use_fp16 ? com.get_local_attn_numel_align<__half>() :
                                                          com.get_local_attn_numel_align<float>();

        torch::Tensor buff = torch::empty(buff_size, float_options);
        torch::Tensor context_layer = torch::zeros({batch_size, seq_len, num_attention_heads * attention_head_size}, float_options);
        torch::Tensor query_layer_list = torch::empty(local_qkv_numel, float_options);
        torch::Tensor key_layer_list = torch::empty(local_qkv_numel, float_options);
        torch::Tensor value_layer_list = torch::empty(local_qkv_numel, float_options);
        torch::Tensor softmax_out_list = torch::empty(local_attn_numel_align, float_options);
        torch::Tensor dropout_out_list = torch::empty(local_attn_numel_align, float_options);
        torch::Tensor dropout_mask_list = do_P_dropout ? torch::empty(local_attn_numel_align, uint8_options) :
                                                         torch::empty(0, float_options);

        if (use_fp16) {
            MMSelfAttnTrainL::forward<__half>(
                    com,
                    (const __half*)query_layer.data_ptr(),
                    (const __half*)key_layer.data_ptr(),
                    (const __half*)value_layer.data_ptr(),
                    (const __half*)local_attention_mask.data_ptr(),
                    (__half*)context_layer.data_ptr(),
                    (__half*)query_layer_list.data_ptr(),
                    (__half*)key_layer_list.data_ptr(),
                    (__half*)value_layer_list.data_ptr(),
                    (__half*)softmax_out_list.data_ptr(),
                    (__half*)dropout_out_list.data_ptr(),
                    (uint8_t*)dropout_mask_list.data_ptr(),
                    (__half*)buff.data_ptr(),
                    batch_size,
                    seq_len,
                    num_attention_heads,
                    attention_head_size,
                    attention_probs_dropout_prob,
                    use_multistream);
        }
        else {
            MMSelfAttnTrainL::forward<float>(
                    com,
                    (const float*)query_layer.data_ptr(),
                    (const float*)key_layer.data_ptr(),
                    (const float*)value_layer.data_ptr(),
                    (const float*)local_attention_mask.data_ptr(),
                    (float*)context_layer.data_ptr(),
                    (float*)query_layer_list.data_ptr(),
                    (float*)key_layer_list.data_ptr(),
                    (float*)value_layer_list.data_ptr(),
                    (float*)softmax_out_list.data_ptr(),
                    (float*)dropout_out_list.data_ptr(),
                    (uint8_t*)dropout_mask_list.data_ptr(),
                    (float*)buff.data_ptr(),
                    batch_size,
                    seq_len,
                    num_attention_heads,
                    attention_head_size,
                    attention_probs_dropout_prob,
                    use_multistream);
        }

        ctx->saved_data["modal_index"] = modal_index;
        ctx->saved_data["query_layer_list"] = query_layer_list;
        ctx->saved_data["key_layer_list"] = key_layer_list;
        ctx->saved_data["value_layer_list"] = value_layer_list;
        ctx->saved_data["softmax_out_list"] = softmax_out_list;
        ctx->saved_data["dropout_out_list"] = dropout_out_list;
        ctx->saved_data["dropout_mask_list"] = dropout_mask_list;
        ctx->saved_data["attention_probs_dropout_prob"] = attention_probs_dropout_prob;
        ctx->saved_data["use_multistream"] = use_multistream;
        ctx->saved_data["num_attention_heads"] = num_attention_heads;
        return context_layer;
    }

    static tensor_list
    backward(torch::autograd::AutogradContext* ctx,
             const tensor_list& grad_list) {
        auto grad_context_layer = grad_list[0];
        CHECK_INPUT(grad_context_layer);
        auto float_options = torch::TensorOptions().dtype(grad_context_layer.dtype()).device(grad_context_layer.device());

        auto modal_index = ctx->saved_data["modal_index"].toTensor();
        auto query_layer_list = ctx->saved_data["query_layer_list"].toTensor();
        auto key_layer_list = ctx->saved_data["key_layer_list"].toTensor();
        auto value_layer_list = ctx->saved_data["value_layer_list"].toTensor();
        auto softmax_out_list = ctx->saved_data["softmax_out_list"].toTensor();
        auto dropout_out_list = ctx->saved_data["dropout_out_list"].toTensor();
        auto dropout_mask_list = ctx->saved_data["dropout_mask_list"].toTensor();
        auto attention_probs_dropout_prob = ctx->saved_data["attention_probs_dropout_prob"].toDouble();
        auto use_multistream = ctx->saved_data["use_multistream"].toBool();
        auto num_attention_heads = ctx->saved_data["num_attention_heads"].toInt();
        ctx->saved_data.clear();

        const int64_t batch_size = grad_context_layer.size(0);
        const int64_t seq_len = grad_context_layer.size(1);
        const int64_t attention_head_size = grad_context_layer.size(2) / num_attention_heads;

        MMSelfAttnCom com((const int64_t*)modal_index.data_ptr(),
                          modal_index.size(0),
                          batch_size,
                          num_attention_heads,
                          attention_head_size);
        const int64_t buff_size = MMSelfAttnTrainL::get_b_buff_size(use_multistream, com);

        torch::Tensor buff = torch::empty(buff_size, float_options);
        torch::Tensor grad_query_layer = torch::zeros({batch_size, seq_len, num_attention_heads, attention_head_size}, float_options);
        torch::Tensor grad_key_layer = torch::zeros({batch_size, seq_len, num_attention_heads, attention_head_size}, float_options);
        torch::Tensor grad_value_layer = torch::zeros({batch_size, seq_len, num_attention_heads, attention_head_size}, float_options);

        if (grad_context_layer.scalar_type() == at::ScalarType::Half) {
            MMSelfAttnTrainL::backward<__half>(
                    com,
                    (const __half*)grad_context_layer.data_ptr(),
                    (const __half*)query_layer_list.data_ptr(),
                    (const __half*)key_layer_list.data_ptr(),
                    (const __half*)value_layer_list.data_ptr(),
                    (const __half*)softmax_out_list.data_ptr(),
                    (__half*)dropout_out_list.data_ptr(),
                    (const uint8_t*)dropout_mask_list.data_ptr(),
                    (__half*)grad_query_layer.data_ptr(),
                    (__half*)grad_key_layer.data_ptr(),
                    (__half*)grad_value_layer.data_ptr(),
                    (__half*)buff.data_ptr(),
                    batch_size,
                    seq_len,
                    num_attention_heads,
                    attention_head_size,
                    attention_probs_dropout_prob,
                    use_multistream);
        }
        else if (grad_context_layer.scalar_type() == at::ScalarType::Float) {
            MMSelfAttnTrainL::backward<float>(
                    com,
                    (const float*)grad_context_layer.data_ptr(),
                    (const float*)query_layer_list.data_ptr(),
                    (const float*)key_layer_list.data_ptr(),
                    (const float*)value_layer_list.data_ptr(),
                    (const float*)softmax_out_list.data_ptr(),
                    (float*)dropout_out_list.data_ptr(),
                    (const uint8_t*)dropout_mask_list.data_ptr(),
                    (float*)grad_query_layer.data_ptr(),
                    (float*)grad_key_layer.data_ptr(),
                    (float*)grad_value_layer.data_ptr(),
                    (float*)buff.data_ptr(),
                    batch_size,
                    seq_len,
                    num_attention_heads,
                    attention_head_size,
                    attention_probs_dropout_prob,
                    use_multistream);
        }
        else {
            errMsg("invalid dtype");
        }

        return {torch::Tensor(),
                grad_query_layer,
                grad_key_layer,
                grad_value_layer,
                torch::Tensor(),
                torch::Tensor(),
                torch::Tensor()};
    }
};


torch::Tensor
MMSelfAttnTrainL(
        const torch::Tensor& modal_index,
        const torch::Tensor& query_layer,
        const torch::Tensor& key_layer,
        const torch::Tensor& value_layer,
        const torch::Tensor& local_attention_mask,
        double attention_probs_dropout_prob,
        bool use_multistream) {
    return MMSelfAttnAutogradL::apply(
            modal_index,
            query_layer,
            key_layer,
            value_layer,
            local_attention_mask,
            attention_probs_dropout_prob,
            use_multistream);
}


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
        bool fast_fp32) {
    CHECK_CONTIGUOUS(modal_index);      // host
    CHECK_INPUT(query_layer);
    CHECK_INPUT(key_layer);
    CHECK_INPUT(value_layer);
    CHECK_INPUT(local_attention_mask);
    CHECK_INPUT(global_k);
    CHECK_INPUT(global_v);
    CHECK_INPUT(global_selection_padding_mask_zeros);
    auto float_options = torch::TensorOptions().dtype(query_layer.dtype()).device(query_layer.device());

    const int64_t modal_cnt = modal_index.size(0);
    const int64_t batch_size = query_layer.size(0);
    const int64_t seq_len = query_layer.size(1);
    const int64_t num_attention_heads = query_layer.size(2);
    const int64_t attention_head_size = query_layer.size(3);
    const int64_t max_num_global_indices_per_batch = global_k.size(2);
    const int64_t global_selection_padding_mask_zeros_nRow = global_selection_padding_mask_zeros.size(0);
    torch::Tensor context_layer = torch::zeros({batch_size, seq_len, num_attention_heads * attention_head_size}, float_options);

    if (query_layer.scalar_type() == at::ScalarType::Half) {
        MMSelfAttnInferGL::forward<__half>(
                at::cuda::getCurrentCUDAStream(),
                (const int64_t*)modal_index.data_ptr(),
                (const __half*)query_layer.data_ptr(),
                (const __half*)key_layer.data_ptr(),
                (const __half*)value_layer.data_ptr(),
                (const __half*)local_attention_mask.data_ptr(),
                (const __half*)global_k.data_ptr(),
                (const __half*)global_v.data_ptr(),
                (const int64_t*)global_selection_padding_mask_zeros.data_ptr(),
                (__half*)context_layer.data_ptr(),
                (__half*)buff.data_ptr(),
                modal_cnt,
                batch_size,
                seq_len,
                num_attention_heads,
                attention_head_size,
                max_num_global_indices_per_batch,
                global_selection_padding_mask_zeros_nRow,
                use_multistream,
                fast_fp32);
    }
    else if (query_layer.scalar_type() == at::ScalarType::Float) {
        MMSelfAttnInferGL::forward<float>(
                at::cuda::getCurrentCUDAStream(),
                (const int64_t*)modal_index.data_ptr(),
                (const float*)query_layer.data_ptr(),
                (const float*)key_layer.data_ptr(),
                (const float*)value_layer.data_ptr(),
                (const float*)local_attention_mask.data_ptr(),
                (const float*)global_k.data_ptr(),
                (const float*)global_v.data_ptr(),
                (const int64_t*)global_selection_padding_mask_zeros.data_ptr(),
                (float*)context_layer.data_ptr(),
                (float*)buff.data_ptr(),
                modal_cnt,
                batch_size,
                seq_len,
                num_attention_heads,
                attention_head_size,
                max_num_global_indices_per_batch,
                global_selection_padding_mask_zeros_nRow,
                use_multistream,
                fast_fp32);
    }
    else {
        errMsg("invalid dtype");
    }

    return context_layer;
}


class MMSelfAttnAutogradGL: public torch::autograd::Function<MMSelfAttnAutogradGL> {
public:
    static torch::Tensor
    forward(torch::autograd::AutogradContext* ctx,
            const torch::Tensor& modal_index,
            const torch::Tensor& query_layer,
            const torch::Tensor& key_layer,
            const torch::Tensor& value_layer,
            const torch::Tensor& local_attention_mask,
            const torch::Tensor& global_k,
            const torch::Tensor& global_v,
            const torch::Tensor& global_selection_padding_mask_zeros,
            double attention_probs_dropout_prob,
            bool use_multistream) {
        CHECK_CONTIGUOUS(modal_index);      // host
        CHECK_INPUT(query_layer);
        CHECK_INPUT(key_layer);
        CHECK_INPUT(value_layer);
        CHECK_INPUT(local_attention_mask);
        CHECK_INPUT(global_k);
        CHECK_INPUT(global_v);
        CHECK_INPUT(global_selection_padding_mask_zeros);
        auto float_options = torch::TensorOptions().dtype(query_layer.dtype()).device(query_layer.device());
        auto uint8_options = torch::TensorOptions().dtype(torch::kUInt8).device(query_layer.device()).requires_grad(false);

        bool use_fp16;
        if (query_layer.scalar_type() == at::ScalarType::Half) {
            use_fp16 = true;
        }
        else if (query_layer.scalar_type() == at::ScalarType::Float) {
            use_fp16 = false;
        }
        else {
            errMsg("invalid dtype");
        }

        const int64_t batch_size = query_layer.size(0);
        const int64_t seq_len = query_layer.size(1);
        const int64_t num_attention_heads = query_layer.size(2);
        const int64_t attention_head_size = query_layer.size(3);
        const int64_t max_num_global_indices_per_batch = global_k.size(2);
        const int64_t global_selection_padding_mask_zeros_nRow = global_selection_padding_mask_zeros.size(0);
        const bool do_P_dropout = isActive(attention_probs_dropout_prob);

        MMSelfAttnCom com((const int64_t*)modal_index.data_ptr(),
                          modal_index.size(0),
                          batch_size,
                          num_attention_heads,
                          attention_head_size,
                          max_num_global_indices_per_batch);
        const int64_t buff_size = use_fp16 ? MMSelfAttnTrainGL::get_f_buff_size<__half>(use_multistream, do_P_dropout, com) :
                                             MMSelfAttnTrainGL::get_f_buff_size<float>(use_multistream, do_P_dropout, com);
        const int64_t local_qkv_numel = com.get_local_qkv_numel();
        const int64_t local_attn_numel_align = use_fp16 ? com.get_local_attn_numel_align<__half>() :
                                                          com.get_local_attn_numel_align<float>();
        const int64_t global_attn_numel_align = use_fp16 ? com.get_global_attn_numel_align<__half>() :
                                                           com.get_global_attn_numel_align<float>();
        const int64_t attn_numel_align = use_fp16 ? com.get_attn_numel_align<__half>() :
                                                    com.get_attn_numel_align<float>();

        torch::Tensor buff = torch::empty(buff_size, float_options);
        torch::Tensor context_layer = torch::zeros({batch_size, seq_len, num_attention_heads * attention_head_size}, float_options);
        torch::Tensor query_layer_list = torch::empty(local_qkv_numel, float_options);
        torch::Tensor key_layer_list = torch::empty(local_qkv_numel, float_options);
        torch::Tensor value_layer_list = torch::empty(local_qkv_numel, float_options);
        torch::Tensor softmax_out_list = torch::empty(attn_numel_align, float_options);
        torch::Tensor dropout_mask_list = do_P_dropout ? torch::empty(attn_numel_align, uint8_options) :
                                                         torch::empty(0, float_options);
        torch::Tensor local_attn_probs_list = torch::empty(local_attn_numel_align, float_options);
        torch::Tensor global_attn_probs_list = torch::empty(global_attn_numel_align, float_options);

        if (use_fp16) {
            MMSelfAttnTrainGL::forward<__half>(
                    com,
                    (const __half*)query_layer.data_ptr(),
                    (const __half*)key_layer.data_ptr(),
                    (const __half*)value_layer.data_ptr(),
                    (const __half*)local_attention_mask.data_ptr(),
                    (const __half*)global_k.data_ptr(),
                    (const __half*)global_v.data_ptr(),
                    (const int64_t*)global_selection_padding_mask_zeros.data_ptr(),
                    (__half*)context_layer.data_ptr(),
                    (__half*)query_layer_list.data_ptr(),
                    (__half*)key_layer_list.data_ptr(),
                    (__half*)value_layer_list.data_ptr(),
                    (__half*)softmax_out_list.data_ptr(),
                    (uint8_t*)dropout_mask_list.data_ptr(),
                    (__half*)local_attn_probs_list.data_ptr(),
                    (__half*)global_attn_probs_list.data_ptr(),
                    (__half*)buff.data_ptr(),
                    batch_size,
                    seq_len,
                    num_attention_heads,
                    attention_head_size,
                    max_num_global_indices_per_batch,
                    global_selection_padding_mask_zeros_nRow,
                    attention_probs_dropout_prob,
                    use_multistream);
        }
        else {
            MMSelfAttnTrainGL::forward<float>(
                    com,
                    (const float*)query_layer.data_ptr(),
                    (const float*)key_layer.data_ptr(),
                    (const float*)value_layer.data_ptr(),
                    (const float*)local_attention_mask.data_ptr(),
                    (const float*)global_k.data_ptr(),
                    (const float*)global_v.data_ptr(),
                    (const int64_t*)global_selection_padding_mask_zeros.data_ptr(),
                    (float*)context_layer.data_ptr(),
                    (float*)query_layer_list.data_ptr(),
                    (float*)key_layer_list.data_ptr(),
                    (float*)value_layer_list.data_ptr(),
                    (float*)softmax_out_list.data_ptr(),
                    (uint8_t*)dropout_mask_list.data_ptr(),
                    (float*)local_attn_probs_list.data_ptr(),
                    (float*)global_attn_probs_list.data_ptr(),
                    (float*)buff.data_ptr(),
                    batch_size,
                    seq_len,
                    num_attention_heads,
                    attention_head_size,
                    max_num_global_indices_per_batch,
                    global_selection_padding_mask_zeros_nRow,
                    attention_probs_dropout_prob,
                    use_multistream);
        }

        ctx->save_for_backward({
                global_k,
                global_v,
                global_selection_padding_mask_zeros});
        ctx->saved_data["modal_index"] = modal_index;
        ctx->saved_data["query_layer_list"] = query_layer_list;
        ctx->saved_data["key_layer_list"] = key_layer_list;
        ctx->saved_data["value_layer_list"] = value_layer_list;
        ctx->saved_data["softmax_out_list"] = softmax_out_list;
        ctx->saved_data["dropout_mask_list"] = dropout_mask_list;
        ctx->saved_data["local_attn_probs_list"] = local_attn_probs_list;
        ctx->saved_data["global_attn_probs_list"] = global_attn_probs_list;
        ctx->saved_data["attention_probs_dropout_prob"] = attention_probs_dropout_prob;
        ctx->saved_data["use_multistream"] = use_multistream;
        ctx->saved_data["num_attention_heads"] = num_attention_heads;
        return context_layer;
    }

    static tensor_list
    backward(torch::autograd::AutogradContext* ctx,
             const tensor_list& grad_list) {
        auto grad_context_layer = grad_list[0];
        CHECK_INPUT(grad_context_layer);
        auto float_options = torch::TensorOptions().dtype(grad_context_layer.dtype()).device(grad_context_layer.device());

        bool use_fp16;
        if (grad_context_layer.scalar_type() == at::ScalarType::Half) {
            use_fp16 = true;
        }
        else if (grad_context_layer.scalar_type() == at::ScalarType::Float) {
            use_fp16 = false;
        }
        else {
            errMsg("invalid dtype");
        }

        tensor_list saved_list = std::move(ctx->get_saved_variables());
        torch::Tensor& global_k = saved_list[0];
        torch::Tensor& global_v = saved_list[1];
        torch::Tensor& global_selection_padding_mask_zeros = saved_list[2];
        auto modal_index = ctx->saved_data["modal_index"].toTensor();
        auto query_layer_list = ctx->saved_data["query_layer_list"].toTensor();
        auto key_layer_list = ctx->saved_data["key_layer_list"].toTensor();
        auto value_layer_list = ctx->saved_data["value_layer_list"].toTensor();
        auto softmax_out_list = ctx->saved_data["softmax_out_list"].toTensor();
        auto dropout_mask_list = ctx->saved_data["dropout_mask_list"].toTensor();
        auto local_attn_probs_list = ctx->saved_data["local_attn_probs_list"].toTensor();
        auto global_attn_probs_list = ctx->saved_data["global_attn_probs_list"].toTensor();
        auto attention_probs_dropout_prob = ctx->saved_data["attention_probs_dropout_prob"].toDouble();
        auto use_multistream = ctx->saved_data["use_multistream"].toBool();
        auto num_attention_heads = ctx->saved_data["num_attention_heads"].toInt();
        ctx->saved_data.clear();

        const int64_t batch_size = grad_context_layer.size(0);
        const int64_t seq_len = grad_context_layer.size(1);
        const int64_t attention_head_size = grad_context_layer.size(2) / num_attention_heads;
        const int64_t max_num_global_indices_per_batch = global_k.size(2);
        const int64_t global_selection_padding_mask_zeros_nRow = global_selection_padding_mask_zeros.size(0);

        MMSelfAttnCom com((const int64_t*)modal_index.data_ptr(),
                          modal_index.size(0),
                          batch_size,
                          num_attention_heads,
                          attention_head_size,
                          max_num_global_indices_per_batch);
        const int64_t buff_size = use_fp16 ? MMSelfAttnTrainGL::get_b_buff_size<__half>(use_multistream, com) :
                                             MMSelfAttnTrainGL::get_b_buff_size<float>(use_multistream, com);

        torch::Tensor buff = torch::empty(buff_size, float_options);
        torch::Tensor grad_query_layer = torch::zeros({batch_size, seq_len, num_attention_heads, attention_head_size}, float_options);
        torch::Tensor grad_key_layer = torch::zeros({batch_size, seq_len, num_attention_heads, attention_head_size}, float_options);
        torch::Tensor grad_value_layer = torch::zeros({batch_size, seq_len, num_attention_heads, attention_head_size}, float_options);
        torch::Tensor grad_global_k = torch::zeros_like(global_k);
        torch::Tensor grad_global_v = torch::zeros_like(global_v);

        if (use_fp16) {
            MMSelfAttnTrainGL::backward<__half>(
                    com,
                    (const __half*)grad_context_layer.data_ptr(),
                    (const __half*)query_layer_list.data_ptr(),
                    (const __half*)key_layer_list.data_ptr(),
                    (__half*)value_layer_list.data_ptr(),
                    (const __half*)global_k.data_ptr(),
                    (const __half*)global_v.data_ptr(),
                    (const int64_t*)global_selection_padding_mask_zeros.data_ptr(),
                    (const __half*)softmax_out_list.data_ptr(),
                    (const uint8_t*)dropout_mask_list.data_ptr(),
                    (__half*)local_attn_probs_list.data_ptr(),
                    (__half*)global_attn_probs_list.data_ptr(),
                    (__half*)grad_query_layer.data_ptr(),
                    (__half*)grad_key_layer.data_ptr(),
                    (__half*)grad_value_layer.data_ptr(),
                    (__half*)grad_global_k.data_ptr(),
                    (__half*)grad_global_v.data_ptr(),
                    (__half*)buff.data_ptr(),
                    batch_size,
                    seq_len,
                    num_attention_heads,
                    attention_head_size,
                    max_num_global_indices_per_batch,
                    global_selection_padding_mask_zeros_nRow,
                    attention_probs_dropout_prob,
                    use_multistream);
        }
        else {
            MMSelfAttnTrainGL::backward<float>(
                    com,
                    (const float*)grad_context_layer.data_ptr(),
                    (const float*)query_layer_list.data_ptr(),
                    (const float*)key_layer_list.data_ptr(),
                    (float*)value_layer_list.data_ptr(),
                    (const float*)global_k.data_ptr(),
                    (const float*)global_v.data_ptr(),
                    (const int64_t*)global_selection_padding_mask_zeros.data_ptr(),
                    (const float*)softmax_out_list.data_ptr(),
                    (const uint8_t*)dropout_mask_list.data_ptr(),
                    (float*)local_attn_probs_list.data_ptr(),
                    (float*)global_attn_probs_list.data_ptr(),
                    (float*)grad_query_layer.data_ptr(),
                    (float*)grad_key_layer.data_ptr(),
                    (float*)grad_value_layer.data_ptr(),
                    (float*)grad_global_k.data_ptr(),
                    (float*)grad_global_v.data_ptr(),
                    (float*)buff.data_ptr(),
                    batch_size,
                    seq_len,
                    num_attention_heads,
                    attention_head_size,
                    max_num_global_indices_per_batch,
                    global_selection_padding_mask_zeros_nRow,
                    attention_probs_dropout_prob,
                    use_multistream);
        }

        return {torch::Tensor(),
                grad_query_layer,
                grad_key_layer,
                grad_value_layer,
                torch::Tensor(),
                grad_global_k,
                grad_global_v,
                torch::Tensor(),
                torch::Tensor(),
                torch::Tensor()};
    }
};


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
        bool use_multistream) {
    return MMSelfAttnAutogradGL::apply(
            modal_index,
            query_layer,
            key_layer,
            value_layer,
            local_attention_mask,
            global_k,
            global_v,
            global_selection_padding_mask_zeros,
            attention_probs_dropout_prob,
            use_multistream);
}

