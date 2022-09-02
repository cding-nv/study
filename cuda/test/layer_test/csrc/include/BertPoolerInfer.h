#pragma once
#include <cuda_runtime.h>


class BertPoolerInfer {
public:
    static int
    get_f_buff_size(int batch_size,
                    int hidden_size,
                    int cls_count);

    template <typename T>
    static void
    forward(cudaStream_t stream,
            const T* hidden_states,
            const T* linear_weight,
            const T* linear_bias,
            T* pooled_output,
            T* buffer,
            int batch_size,
            int seq_len,
            int hidden_size,
            int cls_count,
            bool fast_fp32);
};

