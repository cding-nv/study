#include "MMSelfAttnInferL.h"
#include "BatchGemm.h"
#include "Context.h"
#include "MMSelfAttnCom.h"
#include "Softmax.h"
#include "UniOps.h"


template <typename T>
void
MMSelfAttnInferL::forward(
        cudaStream_t stream0,
        const int64_t* modal_index,
        const T* query_layer,
        const T* key_layer,
        const T* value_layer,
        const T* local_attention_mask,
        T* context_layer,
        T* buffer,
        int modal_cnt,
        int batch_size,
        int seq_len,
        int num_attention_heads,
        int attention_head_size,
        bool use_multistream,
        bool fast_fp32) {
    MMSelfAttnCom com(modal_index, modal_cnt, batch_size, num_attention_heads, attention_head_size);
    const float alpha = 1. / sqrt(attention_head_size);

    for (int i = 0; i < modal_cnt; ++i) {
        cudaStream_t stream = use_multistream ? Context::instance().getStream(i) : stream0;
        cublasHandle_t handle = Context::instance().getHandle(stream, fast_fp32);

        const int begin = com.get_index_begin(i);
        const int cur_seq_len = com.get_current_seq_len(i);
        const int cur_l_qkv_numel = com.get_current_local_qkv_numel(i);
        const int cur_l_attn_numel_align = com.get_current_local_attn_numel_align<T>(i);

        T* buff_A = buffer;                                         // cur_l_qkv_numel
        T* buff_B = buffer + cur_l_qkv_numel;                       // cur_l_qkv_numel
        T* buff_C = buffer + cur_l_qkv_numel * 2;                   // cur_l_attn_numel_align
        if (use_multistream) {
            buffer += cur_l_qkv_numel * 2 + cur_l_attn_numel_align;
        }

        SlicePermute<T>::run(
                begin,
                seq_len,
                query_layer,
                batch_size,
                num_attention_heads,
                cur_seq_len,
                attention_head_size,
                buff_A,                             // A0, cur_l_qkv_numel, out
                stream);

        SlicePermute<T>::run(
                begin,
                seq_len,
                key_layer,
                batch_size,
                num_attention_heads,
                cur_seq_len,
                attention_head_size,
                buff_B,                             // B0, cur_l_qkv_numel, out
                stream);

        SliceExpand12<T>::run(
                begin,
                seq_len,
                local_attention_mask,
                batch_size,
                num_attention_heads,
                cur_seq_len,
                cur_seq_len,
                buff_C,                             // cur_l_attn_numel_align, out
                stream);

        BatchGemm<T, false, true>::forwardA(
                buff_A,                             // A0, cur_l_qkv_numel, in
                buff_B,                             // B0, cur_l_qkv_numel, in
                buff_C,                             // cur_l_attn_numel_align, inout
                handle,
                batch_size * num_attention_heads,
                cur_seq_len,
                cur_seq_len,
                attention_head_size,
                alpha);

        Softmax<T>::forward(
                buff_C,                             // cur_l_attn_numel_align, inout
                batch_size * num_attention_heads * cur_seq_len,
                cur_seq_len,
                stream);

        SlicePermute<T>::run(
                begin,
                seq_len,
                value_layer,
                batch_size,
                num_attention_heads,
                cur_seq_len,
                attention_head_size,
                buff_A,                             // A1, cur_l_qkv_numel, out
                stream);

        BatchGemm<T, false, false>::forward(
                buff_C,                             // cur_l_attn_numel_align, in
                buff_A,                             // A1, cur_l_qkv_numel, in
                buff_B,                             // B1, cur_l_qkv_numel, out
                handle,
                batch_size * num_attention_heads,
                cur_seq_len,
                attention_head_size,
                cur_seq_len);

        PermuteSlice<T>::run(
                begin,
                seq_len,
                context_layer,                      // local_qkv_numel, out
                batch_size,
                num_attention_heads,
                cur_seq_len,
                attention_head_size,
                buff_B,                             // B1, cur_l_qkv_numel, in
                stream);
    }
}


template
void
MMSelfAttnInferL::forward(
        cudaStream_t stream0,
        const int64_t* modal_index,
        const float* query_layer,
        const float* key_layer,
        const float* value_layer,
        const float* local_attention_mask,
        float* context_layer,
        float* buffer,
        int modal_cnt,
        int batch_size,
        int seq_len,
        int num_attention_heads,
        int attention_head_size,
        bool use_multistream,
        bool fast_fp32);


template
void
MMSelfAttnInferL::forward(
        cudaStream_t stream0,
        const int64_t* modal_index,
        const __half* query_layer,
        const __half* key_layer,
        const __half* value_layer,
        const __half* local_attention_mask,
        __half* context_layer,
        __half* buffer,
        int modal_cnt,
        int batch_size,
        int seq_len,
        int num_attention_heads,
        int attention_head_size,
        bool use_multistream,
        bool fast_fp32);

