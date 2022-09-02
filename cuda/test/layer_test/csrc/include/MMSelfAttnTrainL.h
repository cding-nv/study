#pragma once
#include "BatchGemm.h"
#include "Dropout.h"
#include "MMSelfAttnCom.h"
#include "Softmax.h"
#include "UniOps.h"
#include "utils.h"


class MMSelfAttnTrainL {
public:
    static int64_t
    get_f_buff_size(bool use_multistream,
                    const MMSelfAttnCom& com) {
        int64_t f_buff_size = use_multistream ? com.get_local_qkv_numel() :
                                                com.get_local_qkv_numel_max();
        return f_buff_size;
    }

    static int64_t
    get_b_buff_size(bool use_multistream,
                    const MMSelfAttnCom& com) {
        int64_t b_buff_size = use_multistream ? 2 * com.get_local_qkv_numel() :
                                                2 * com.get_local_qkv_numel_max();
        return b_buff_size;
    }

    template <typename T>
    static void
    forward(const MMSelfAttnCom& com,
            const T* query_layer,
            const T* key_layer,
            const T* value_layer,
            const T* local_attention_mask,
            T* context_layer,
            T* query_layer_list,
            T* key_layer_list,
            T* value_layer_list,
            T* softmax_out_list,
            T* dropout_out_list,
            uint8_t* dropout_mask_list,             // do_P_dropout
            T* buffer,
            int64_t batch_size,
            int64_t seq_len,
            int64_t num_attention_heads,
            int64_t attention_head_size,
            double attention_probs_dropout_prob,
            bool use_multistream) {
        const float alpha = 1. / sqrt(attention_head_size);

        int64_t qkv_offset = 0LL;
        int64_t local_attn_offset = 0LL;
        for (int64_t i = 0LL; i < com.get_modal_cnt(); ++i) {
            cudaStream_t stream = Context::instance().getStream(i * use_multistream);
            cublasHandle_t handle = Context::instance().getHandle(stream);

            const int64_t begin = com.get_index_begin(i);
            const int64_t cur_seq_len = com.get_current_seq_len(i);
            const int64_t cur_l_qkv_numel = com.get_current_local_qkv_numel(i);
            const int64_t cur_l_attn_numel_align = com.get_current_local_attn_numel_align<T>(i);

            T* cur_query_layer = query_layer_list + qkv_offset;
            T* cur_key_layer = key_layer_list + qkv_offset;
            T* cur_value_layer = value_layer_list + qkv_offset;
            T* cur_softmax_out = softmax_out_list + local_attn_offset;
            T* cur_dropout_out = dropout_out_list + local_attn_offset;
            uint8_t* cur_dropout_mask = dropout_mask_list + local_attn_offset;
            T* buff_A = buffer;                     // cur_l_qkv_numel

            qkv_offset += cur_l_qkv_numel;
            local_attn_offset += cur_l_attn_numel_align;
            if (use_multistream) {
                buffer += cur_l_qkv_numel;
            }

            SlicePermute3<T>::run(
                    begin,
                    seq_len,
                    query_layer,
                    key_layer,
                    value_layer,
                    batch_size,
                    num_attention_heads,
                    cur_seq_len,
                    attention_head_size,
                    cur_query_layer,                // cur_l_qkv_numel, out
                    cur_key_layer,                  // cur_l_qkv_numel, out
                    cur_value_layer,                // cur_l_qkv_numel, out
                    stream);

            SliceExpand12<T>::run(
                    begin,
                    seq_len,
                    local_attention_mask,
                    batch_size,
                    num_attention_heads,
                    cur_seq_len,
                    cur_seq_len,
                    cur_softmax_out,                // cur_l_attn_numel_align, out
                    stream);

            BatchGemm<T, false, true>::forwardA(
                    cur_query_layer,                // cur_l_qkv_numel, in
                    cur_key_layer,                  // cur_l_qkv_numel, in
                    cur_softmax_out,                // cur_l_attn_numel_align, inout
                    handle,
                    batch_size * num_attention_heads,
                    cur_seq_len,
                    cur_seq_len,
                    attention_head_size,
                    alpha);

            Softmax<T>::forward(
                    cur_softmax_out,                // cur_l_attn_numel_align, inout
                    batch_size * num_attention_heads * cur_seq_len,
                    cur_seq_len,
                    stream);

            if (isActive(attention_probs_dropout_prob)) {
                Dropout<T>::forward(
                        cur_softmax_out,            // cur_l_attn_numel_align, in
                        cur_dropout_out,            // cur_l_attn_numel_align, out
                        cur_dropout_mask,           // cur_l_attn_numel_align, out
                        attention_probs_dropout_prob,
                        toScale(attention_probs_dropout_prob),
                        com.get_current_attn_numel(i),
                        stream);

                BatchGemm<T, false, false>::forward(
                        cur_dropout_out,            // cur_l_attn_numel_align, in
                        cur_value_layer,            // cur_l_qkv_numel, in
                        buff_A,                     // cur_l_qkv_numel, out
                        handle,
                        batch_size * num_attention_heads,
                        cur_seq_len,
                        attention_head_size,
                        cur_seq_len);
            }
            else {
                BatchGemm<T, false, false>::forward(
                        cur_softmax_out,            // cur_l_attn_numel_align, in
                        cur_value_layer,            // cur_l_qkv_numel, in
                        buff_A,                     // cur_l_qkv_numel, out
                        handle,
                        batch_size * num_attention_heads,
                        cur_seq_len,
                        attention_head_size,
                        cur_seq_len);
            }

            PermuteSlice<T>::run(
                    begin,
                    seq_len,
                    context_layer,                  // local_qkv_numel, out
                    batch_size,
                    num_attention_heads,
                    cur_seq_len,
                    attention_head_size,
                    buff_A,                         // cur_l_qkv_numel, in
                    stream);
        }
    }

    template <typename T>
    static void
    backward(const MMSelfAttnCom& com,
             const T* grad_context_layer,
             const T* query_layer_list,
             const T* key_layer_list,
             const T* value_layer_list,
             const T* softmax_out_list,
             T* dropout_out_list,                   // in, buff
             const uint8_t* dropout_mask_list,      // do_P_dropout
             T* grad_query_layer,
             T* grad_key_layer,
             T* grad_value_layer,
             T* buffer,
             int64_t batch_size,
             int64_t seq_len,
             int64_t num_attention_heads,
             int64_t attention_head_size,
             double attention_probs_dropout_prob,
             bool use_multistream) {
        const float alpha = 1. / sqrt(attention_head_size);

        int64_t qkv_offset = 0LL;
        int64_t local_attn_offset = 0LL;
        for (int64_t i = 0LL; i < com.get_modal_cnt(); ++i) {
            cudaStream_t stream = Context::instance().getStream(i * use_multistream);
            cublasHandle_t handle = Context::instance().getHandle(stream);

            const int64_t begin = com.get_index_begin(i);
            const int64_t cur_seq_len = com.get_current_seq_len(i);
            const int64_t cur_l_qkv_numel = com.get_current_local_qkv_numel(i);
            const int64_t cur_l_attn_numel_align = com.get_current_local_attn_numel_align<T>(i);

            const T* cur_query_layer = query_layer_list + qkv_offset;
            const T* cur_key_layer = key_layer_list + qkv_offset;
            const T* cur_value_layer = value_layer_list + qkv_offset;
            const T* cur_softmax_out = softmax_out_list + local_attn_offset;
            T* cur_dropout_out = dropout_out_list + local_attn_offset;
            const uint8_t* cur_dropout_mask = dropout_mask_list + local_attn_offset;

            T* buff_B = buffer;                     // cur_l_qkv_numel
            T* buff_C = buffer + cur_l_qkv_numel;   // cur_l_qkv_numel
            T* buff_D = cur_dropout_out;            // cur_l_attn_numel_align

            qkv_offset += cur_l_qkv_numel;
            local_attn_offset += cur_l_attn_numel_align;
            if (use_multistream) {
                buffer += cur_l_qkv_numel * 2;
            }

            SlicePermute<T>::run(
                    begin,
                    seq_len,
                    grad_context_layer,
                    batch_size,
                    num_attention_heads,
                    cur_seq_len,
                    attention_head_size,
                    buff_B,                         // B0, cur_l_qkv_numel, out
                    stream);

            if (isActive(attention_probs_dropout_prob)) {
                BatchGemm<T, false, false>::backward(
                        buff_B,                     // B0, cur_l_qkv_numel, in
                        cur_dropout_out,            // cur_l_attn_numel_align, in
                        cur_value_layer,            // cur_l_qkv_numel, in
                        buff_D,                     // cur_l_attn_numel_align, out
                        buff_C,                     // C0, cur_l_qkv_numel, out
                        handle,
                        batch_size * num_attention_heads,
                        cur_seq_len,
                        attention_head_size,
                        cur_seq_len);

                Dropout<T>::backward(
                        buff_D,                     // cur_l_attn_numel_align, inout
                        cur_dropout_mask,           // cur_l_attn_numel_align, in
                        toScale(attention_probs_dropout_prob),
                        com.get_current_attn_numel(i),
                        stream);
            }
            else {
                BatchGemm<T, false, false>::backward(
                        buff_B,                     // B0, cur_l_qkv_numel, in
                        cur_softmax_out,            // cur_l_attn_numel_align, in
                        cur_value_layer,            // cur_l_qkv_numel, in
                        buff_D,                     // cur_l_attn_numel_align, out
                        buff_C,                     // C0, cur_l_qkv_numel, out
                        handle,
                        batch_size * num_attention_heads,
                        cur_seq_len,
                        attention_head_size,
                        cur_seq_len);
            }

            PermuteSlice<T>::run(
                    begin,
                    seq_len,
                    grad_value_layer,               // local_qkv_numel, out
                    batch_size,
                    num_attention_heads,
                    cur_seq_len,
                    attention_head_size,
                    buff_C,                         // C0, cur_l_qkv_numel, in
                    stream);

            Softmax<T>::backward(
                    cur_softmax_out,                // cur_l_attn_numel_align, in
                    buff_D,                         // cur_l_attn_numel_align, inout
                    batch_size * num_attention_heads * cur_seq_len,
                    cur_seq_len,
                    stream);

            BatchGemm<T, false, true>::backward(
                    buff_D,                         // cur_l_attn_numel_align, in
                    cur_query_layer,                // cur_l_qkv_numel, in
                    cur_key_layer,                  // cur_l_qkv_numel, in
                    buff_B,                         // B1, cur_l_qkv_numel, out
                    buff_C,                         // C1, cur_l_qkv_numel, out
                    handle,
                    batch_size * num_attention_heads,
                    cur_seq_len,
                    cur_seq_len,
                    attention_head_size,
                    alpha);

            PermuteSlice<T>::run(
                    begin,
                    seq_len,
                    grad_query_layer,               // local_qkv_numel, out
                    batch_size,
                    num_attention_heads,
                    cur_seq_len,
                    attention_head_size,
                    buff_B,                         // B1, cur_l_qkv_numel, in
                    stream);

            PermuteSlice<T>::run(
                    begin,
                    seq_len,
                    grad_key_layer,                 // local_qkv_numel, out
                    batch_size,
                    num_attention_heads,
                    cur_seq_len,
                    attention_head_size,
                    buff_C,                         // C1, cur_l_qkv_numel, in
                    stream);
        }
    }
};

