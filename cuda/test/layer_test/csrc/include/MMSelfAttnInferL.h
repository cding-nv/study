#pragma once
#include <cuda_runtime.h>


class MMSelfAttnInferL {
public:
    template <typename T>
    static void
    forward(cudaStream_t stream0,
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
            bool fast_fp32);
};

