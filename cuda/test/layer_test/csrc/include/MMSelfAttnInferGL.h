#pragma once
#include <cuda_runtime.h>


class MMSelfAttnInferGL {
public:
    template <typename T>
    static void
    forward(cudaStream_t stream0,
            const int64_t* modal_index,
            const T* query_layer,
            const T* key_layer,
            const T* value_layer,
            const T* local_attention_mask,
            const T* global_k,
            const T* global_v,
            const int64_t* global_selection_padding_mask_zeros,
            T* context_layer,
            T* buffer,
            int modal_cnt,
            int batch_size,
            int seq_len,
            int num_attention_heads,
            int attention_head_size,
            int max_num_global_indices_per_batch,
            int global_selection_padding_mask_zeros_nRow,
            bool use_multistream,
            bool fast_fp32);
};

