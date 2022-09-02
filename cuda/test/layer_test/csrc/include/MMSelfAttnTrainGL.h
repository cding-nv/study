#pragma once
#include "BatchGemm.h"
#include "Dropout.h"
#include "MMSelfAttnCom.h"
#include "Softmax.h"
#include "UniOps.h"
#include "utils.h"


class MMSelfAttnTrainGL {
public:
    template <typename T>
    static int64_t
    get_f_buff_size(bool use_multistream,
                    bool do_P_dropout,
                    const MMSelfAttnCom& com) {
        if (use_multistream) {
            if (do_P_dropout) {
                int64_t f_buff_size = 0LL;
                for (int64_t i = 0LL; i < com.get_modal_cnt(); ++i) {
                    int64_t cur_l_qkv_numel = com.get_current_local_qkv_numel(i);
                    int64_t cur_attn_numel_align = com.get_current_attn_numel_align<T>(i);
                    int64_t cur_val = max(cur_l_qkv_numel, cur_attn_numel_align);
                    f_buff_size += cur_val;
                }
                return f_buff_size;
            }
            else {
                int64_t f_buff_size = com.get_local_qkv_numel();
                return f_buff_size;
            }
        }
        else {
            if (do_P_dropout) {
                int64_t f_buff_size = 0LL;
                for (int64_t i = 0LL; i < com.get_modal_cnt(); ++i) {
                    int64_t cur_l_qkv_numel = com.get_current_local_qkv_numel(i);
                    int64_t cur_attn_numel_align = com.get_current_attn_numel_align<T>(i);
                    int64_t cur_val = max(cur_l_qkv_numel, cur_attn_numel_align);
                    f_buff_size = max(f_buff_size, cur_val);
                }
                return f_buff_size;
            }
            else {
                int64_t f_buff_size = com.get_local_qkv_numel_max();
                return f_buff_size;
            }
        }
    }

    template <typename T>
    static int64_t
    get_b_buff_size(bool use_multistream,
                    const MMSelfAttnCom& com) {
        if (use_multistream) {
            int64_t b_buff_size = 0LL;
            for (int64_t i = 0LL; i < com.get_modal_cnt(); ++i) {
                int64_t cur_l_qkv_numel = com.get_current_local_qkv_numel(i);
                int64_t cur_attn_numel_align = com.get_current_attn_numel_align<T>(i);
                int64_t global_qkv_numel = com.get_global_qkv_numel();
                int64_t cur_val = max(cur_l_qkv_numel, cur_attn_numel_align) + max(cur_l_qkv_numel, global_qkv_numel);
                b_buff_size += cur_val;
            }
            return b_buff_size;
        }
        else {
            int64_t b_buff_size = 0LL;
            for (int64_t i = 0LL; i < com.get_modal_cnt(); ++i) {
                int64_t cur_l_qkv_numel = com.get_current_local_qkv_numel(i);
                int64_t cur_attn_numel_align = com.get_current_attn_numel_align<T>(i);
                int64_t global_qkv_numel = com.get_global_qkv_numel();
                int64_t cur_val = max(cur_l_qkv_numel, cur_attn_numel_align) + max(cur_l_qkv_numel, global_qkv_numel);
                b_buff_size = max(b_buff_size, cur_val);
            }
            return b_buff_size;
        }
    }

    template <typename T>
    static void
    forward(const MMSelfAttnCom& com,
            const T* query_layer,
            const T* key_layer,
            const T* value_layer,
            const T* local_attention_mask,
            const T* global_k,
            const T* global_v,
            const int64_t* global_selection_padding_mask_zeros,
            T* context_layer,
            T* query_layer_list,
            T* key_layer_list,
            T* value_layer_list,
            T* softmax_out_list,
            uint8_t* dropout_mask_list,             // do_P_dropout
            T* local_attn_probs_list,
            T* global_attn_probs_list,
            T* buffer,
            int64_t batch_size,
            int64_t seq_len,
            int64_t num_attention_heads,
            int64_t attention_head_size,
            int64_t max_num_global_indices_per_batch,
            int64_t global_selection_padding_mask_zeros_nRow,
            double attention_probs_dropout_prob,
            bool use_multistream) {
        const float alpha = 1. / sqrt(attention_head_size);
        const bool do_P_dropout = isActive(attention_probs_dropout_prob);

        int64_t qkv_offset = 0LL;
        int64_t local_attn_offset = 0LL;
        int64_t global_attn_offset = 0LL;
        int64_t attn_offset = 0LL;
        for (int64_t i = 0LL; i < com.get_modal_cnt(); ++i) {
            cudaStream_t stream = Context::instance().getStream(i * use_multistream);
            cublasHandle_t handle = Context::instance().getHandle(stream);

            const int64_t begin = com.get_index_begin(i);
            const int64_t cur_seq_len = com.get_current_seq_len(i);
            const int64_t cur_l_qkv_numel = com.get_current_local_qkv_numel(i);
            const int64_t cur_l_attn_numel_align = com.get_current_local_attn_numel_align<T>(i);
            const int64_t cur_g_attn_numel_align = com.get_current_global_attn_numel_align<T>(i);
            const int64_t cur_attn_numel_align = com.get_current_attn_numel_align<T>(i);

            T* cur_query_layer = query_layer_list + qkv_offset;
            T* cur_key_layer = key_layer_list + qkv_offset;
            T* cur_value_layer = value_layer_list + qkv_offset;
            T* cur_softmax_out = softmax_out_list + attn_offset;
            uint8_t* cur_dropout_mask = dropout_mask_list + attn_offset;
            T* cur_l_attn_probs = local_attn_probs_list + local_attn_offset;
            T* cur_g_attn_probs = global_attn_probs_list + global_attn_offset;

            T* buff_A = cur_l_attn_probs;           // cur_l_attn_numel_align
            T* buff_B = cur_g_attn_probs;           // cur_g_attn_numel_align
            T* buff_C = buffer;                     // do_P_dropout ? max(cur_l_qkv_numel, cur_attn_numel_align) : cur_l_qkv_numel

            qkv_offset += cur_l_qkv_numel;
            local_attn_offset += cur_l_attn_numel_align;
            global_attn_offset += cur_g_attn_numel_align;
            attn_offset += cur_attn_numel_align;
            if (use_multistream) {
                buffer += do_P_dropout ? max(cur_l_qkv_numel, cur_attn_numel_align) : cur_l_qkv_numel;
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
                    buff_A,                         // cur_l_attn_numel_align, out
                    stream);

            BatchGemm<T, false, true>::forwardA(
                    cur_query_layer,                // cur_l_qkv_numel, in
                    cur_key_layer,                  // cur_l_qkv_numel, in
                    buff_A,                         // cur_l_attn_numel_align, inout
                    handle,
                    batch_size * num_attention_heads,
                    cur_seq_len,
                    cur_seq_len,
                    attention_head_size,
                    alpha);

            BatchGemm<T, false, true>::forward(
                    cur_query_layer,                // cur_l_qkv_numel, in
                    global_k,
                    buff_B,                         // cur_g_attn_numel_align, out
                    handle,
                    batch_size * num_attention_heads,
                    cur_seq_len,
                    max_num_global_indices_per_batch,
                    attention_head_size,
                    alpha);

            if (global_selection_padding_mask_zeros_nRow) {
                CopySlice<T>::run(
                        num_attention_heads,
                        cur_seq_len,
                        max_num_global_indices_per_batch,
                        buff_B,                     // cur_g_attn_numel_align, inout
                        global_selection_padding_mask_zeros_nRow,
                        global_selection_padding_mask_zeros,
                        -10000.f,
                        stream);
            }

            Cat<T>::run(
                    batch_size * num_attention_heads * cur_seq_len,
                    max_num_global_indices_per_batch,
                    buff_B,                         // cur_g_attn_numel_align, in
                    cur_seq_len,
                    buff_A,                         // cur_l_attn_numel_align, in
                    cur_softmax_out,                // cur_attn_numel_align, out
                    stream);

            Softmax<T>::forward(
                    cur_softmax_out,                // cur_attn_numel_align, inout
                    batch_size * num_attention_heads * cur_seq_len,
                    max_num_global_indices_per_batch + cur_seq_len,
                    stream);

            if (do_P_dropout) {
                Dropout<T>::forward(
                        cur_softmax_out,            // cur_attn_numel_align, in
                        buff_C,                     // C0, cur_attn_numel_align, out
                        cur_dropout_mask,           // cur_attn_numel_align, out
                        attention_probs_dropout_prob,
                        toScale(attention_probs_dropout_prob),
                        com.get_current_attn_numel(i),
                        stream);

                Narrow<T>::run(
                        batch_size * num_attention_heads * cur_seq_len,
                        buff_C,                     // C0, cur_attn_numel_align, in
                        max_num_global_indices_per_batch,
                        cur_g_attn_probs,           // cur_g_attn_numel_align, out
                        cur_seq_len,
                        cur_l_attn_probs,           // cur_l_attn_numel_align, out
                        stream);
            }
            else {
                Narrow<T>::run(
                        batch_size * num_attention_heads * cur_seq_len,
                        cur_softmax_out,            // cur_attn_numel_align, in
                        max_num_global_indices_per_batch,
                        cur_g_attn_probs,           // cur_g_attn_numel_align, out
                        cur_seq_len,
                        cur_l_attn_probs,           // cur_l_attn_numel_align, out
                        stream);
            }

            BatchGemm<T, false, false>::forward(
                    cur_g_attn_probs,               // cur_g_attn_numel_align, in
                    global_v,
                    buff_C,                         // C1, cur_l_qkv_numel, out
                    handle,
                    batch_size * num_attention_heads,
                    cur_seq_len,
                    attention_head_size,
                    max_num_global_indices_per_batch);

            BatchGemm<T, false, false>::forwardA(
                    cur_l_attn_probs,               // cur_l_attn_numel_align, in
                    cur_value_layer,                // cur_l_qkv_numel, in
                    buff_C,                         // C1, cur_l_qkv_numel, inout
                    handle,
                    batch_size * num_attention_heads,
                    cur_seq_len,
                    attention_head_size,
                    cur_seq_len);

            PermuteSlice<T>::run(
                    begin,
                    seq_len,
                    context_layer,                  // local_qkv_numel, out
                    batch_size,
                    num_attention_heads,
                    cur_seq_len,
                    attention_head_size,
                    buff_C,                         // C1, cur_l_qkv_numel, in
                    stream);
        }
    }

    template <typename T>
    static void
    backward(const MMSelfAttnCom& com,
             const T* grad_context_layer,
             const T* query_layer_list,
             const T* key_layer_list,
             T* value_layer_list,                   // in, buff
             const T* global_k,
             const T* global_v,
             const int64_t* global_selection_padding_mask_zeros,
             const T* softmax_out_list,
             const uint8_t* dropout_mask_list,      // do_P_dropout
             T* local_attn_probs_list,              // in, buff
             T* global_attn_probs_list,             // in, buff
             T* grad_query_layer,
             T* grad_key_layer,
             T* grad_value_layer,
             T* grad_global_k,
             T* grad_global_v,
             T* buffer,
             int64_t batch_size,
             int64_t seq_len,
             int64_t num_attention_heads,
             int64_t attention_head_size,
             int64_t max_num_global_indices_per_batch,
             int64_t global_selection_padding_mask_zeros_nRow,
             double attention_probs_dropout_prob,
             bool use_multistream) {
        const int64_t global_qkv_numel = com.get_global_qkv_numel();
        const float alpha = 1. / sqrt(attention_head_size);

        int64_t qkv_offset = 0LL;
        int64_t local_attn_offset = 0LL;
        int64_t global_attn_offset = 0LL;
        int64_t attn_offset = 0LL;
        for (int64_t i = 0LL; i < com.get_modal_cnt(); ++i) {
            cudaStream_t stream = Context::instance().getStream(i * use_multistream);
            cublasHandle_t handle = Context::instance().getHandle(stream);

            const int64_t begin = com.get_index_begin(i);
            const int64_t cur_seq_len = com.get_current_seq_len(i);
            const int64_t cur_l_qkv_numel = com.get_current_local_qkv_numel(i);
            const int64_t cur_l_attn_numel_align = com.get_current_local_attn_numel_align<T>(i);
            const int64_t cur_g_attn_numel_align = com.get_current_global_attn_numel_align<T>(i);
            const int64_t cur_attn_numel_align = com.get_current_attn_numel_align<T>(i);

            const T* cur_query_layer = query_layer_list + qkv_offset;
            const T* cur_key_layer = key_layer_list + qkv_offset;
            T* cur_value_layer = value_layer_list + qkv_offset;
            const T* cur_softmax_out = softmax_out_list + attn_offset;
            const uint8_t* cur_dropout_mask = dropout_mask_list + attn_offset;
            T* cur_l_attn_probs = local_attn_probs_list + local_attn_offset;
            T* cur_g_attn_probs = global_attn_probs_list + global_attn_offset;

            T* buff_D = cur_l_attn_probs;           // cur_l_attn_numel_align
            T* buff_E = cur_g_attn_probs;           // cur_g_attn_numel_align
            T* buff_F = cur_value_layer;            // cur_l_qkv_numel
            T* buff_G = buffer;                     // max(cur_l_qkv_numel, cur_attn_numel_align)
            T* buff_H = buffer + max(cur_l_qkv_numel, cur_attn_numel_align);    // max(cur_l_qkv_numel, global_qkv_numel)

            qkv_offset += cur_l_qkv_numel;
            local_attn_offset += cur_l_attn_numel_align;
            global_attn_offset += cur_g_attn_numel_align;
            attn_offset += cur_attn_numel_align;
            if (use_multistream) {
                buffer += max(cur_l_qkv_numel, cur_attn_numel_align) + max(cur_l_qkv_numel, global_qkv_numel);
            }

            SlicePermute<T>::run(
                    begin,
                    seq_len,
                    grad_context_layer,
                    batch_size,
                    num_attention_heads,
                    cur_seq_len,
                    attention_head_size,
                    buff_G,                         // G0, cur_l_qkv_numel, out
                    stream);

            BatchGemm<T, false, false>::backward(
                    buff_G,                         // G0, cur_l_qkv_numel, in
                    cur_l_attn_probs,               // cur_l_attn_numel_align, in
                    cur_value_layer,                // cur_l_qkv_numel, in
                    buff_D,                         // D0, cur_l_attn_numel_align, out
                    buff_H,                         // H0, cur_l_qkv_numel, out
                    handle,
                    batch_size * num_attention_heads,
                    cur_seq_len,
                    attention_head_size,
                    cur_seq_len);

            PermuteSlice<T>::run(
                    begin,
                    seq_len,
                    grad_value_layer,               // local_qkv_numel, out
                    batch_size,
                    num_attention_heads,
                    cur_seq_len,
                    attention_head_size,
                    buff_H,                         // H0, cur_l_qkv_numel, in
                    stream);

            BatchGemm<T, false, false>::backward(
                    buff_G,                         // G0, cur_l_qkv_numel, in
                    cur_g_attn_probs,               // cur_g_attn_numel_align, in
                    global_v,
                    buff_E,                         // E0, cur_g_attn_numel_align, out
                    buff_H,                         // H1, global_qkv_numel, out
                    handle,
                    batch_size * num_attention_heads,
                    cur_seq_len,
                    attention_head_size,
                    max_num_global_indices_per_batch);

            AtomicAdd<T>::run(
                    buff_H,                         // H1, global_qkv_numel, in
                    grad_global_v,                  // global_qkv_numel, out
                    global_qkv_numel,
                    stream);

            Cat<T>::run(
                    batch_size * num_attention_heads * cur_seq_len,
                    max_num_global_indices_per_batch,
                    buff_E,                         // E0, cur_g_attn_numel_align, in
                    cur_seq_len,
                    buff_D,                         // D0, cur_l_attn_numel_align, in
                    buff_G,                         // G1, cur_attn_numel_align, out
                    stream);

            if (isActive(attention_probs_dropout_prob)) {
                Dropout<T>::backward(
                        buff_G,                     // G1, cur_attn_numel_align, inout
                        cur_dropout_mask,           // cur_attn_numel_align, in
                        toScale(attention_probs_dropout_prob),
                        com.get_current_attn_numel(i),
                        stream);
            }

            Softmax<T>::backward(
                    cur_softmax_out,                // cur_attn_numel_align, in
                    buff_G,                         // G1, cur_attn_numel_align, inout
                    batch_size * num_attention_heads * cur_seq_len,
                    max_num_global_indices_per_batch + cur_seq_len,
                    stream);

            Narrow<T>::run(
                    batch_size * num_attention_heads * cur_seq_len,
                    buff_G,                         // G1, cur_attn_numel_align, in
                    max_num_global_indices_per_batch,
                    buff_E,                         // E1, cur_g_attn_numel_align, out
                    cur_seq_len,
                    buff_D,                         // D1, cur_l_attn_numel_align, out
                    stream);

            if (global_selection_padding_mask_zeros_nRow) {
                CopySlice<T>::run(
                        num_attention_heads,
                        cur_seq_len,
                        max_num_global_indices_per_batch,
                        buff_E,                     // E1, cur_g_attn_numel_align, inout
                        global_selection_padding_mask_zeros_nRow,
                        global_selection_padding_mask_zeros,
                        0.f,
                        stream);
            }

            BatchGemm<T, false, true>::backward(
                    buff_E,                         // E1, cur_g_attn_numel_align, in
                    cur_query_layer,                // cur_l_qkv_numel, in
                    global_k,
                    buff_F,                         // cur_l_qkv_numel, out
                    buff_H,                         // H2, global_qkv_numel, out
                    handle,
                    batch_size * num_attention_heads,
                    cur_seq_len,
                    max_num_global_indices_per_batch,
                    attention_head_size,
                    alpha);

            AtomicAdd<T>::run(
                    buff_H,                         // H2, global_qkv_numel, in
                    grad_global_k,                  // global_qkv_numel, out
                    global_qkv_numel,
                    stream);

            BatchGemm<T, false, true>::backwardA(
                    buff_D,                         // D1, cur_l_attn_numel_align, in
                    cur_query_layer,                // cur_l_qkv_numel, in
                    cur_key_layer,                  // cur_l_qkv_numel, in
                    buff_F,                         // cur_l_qkv_numel, inout
                    buff_G,                         // G2, cur_l_qkv_numel, out
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
                    buff_F,                         // cur_l_qkv_numel, in
                    stream);

            PermuteSlice<T>::run(
                    begin,
                    seq_len,
                    grad_key_layer,                 // local_qkv_numel, out
                    batch_size,
                    num_attention_heads,
                    cur_seq_len,
                    attention_head_size,
                    buff_G,                         // G2, cur_l_qkv_numel, in
                    stream);
        }
    }
};

