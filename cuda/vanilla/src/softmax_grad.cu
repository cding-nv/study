/*
* Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
/**
* Open sourced multi-head attention
**/

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include "common_op.h"
#include "grad_op.h"
#include "general_kernels.h"

namespace multiheadattention{

namespace softmaxbackprop{

template<OperationType OpType_>
void SoftmaxGrad<OpType_>::softmaxBack_kernelLauncher(
      cudaStream_t stream,
      const DataType_* grads, // input
      const DataType_* softmax, // input
      const bool* mask, // input
      const int batch,
      const int from_seq_len,
      const int to_seq_len,
      const int head_num,
      const int size_per_head)
{

    // TODO: T should be less than 1024
    // TODO: scalar can be computed on CPU
    if (OpType_ == OperationType::FP32) {
        launch_backprop_masking_softmax<float>((float*)param_.grads, (float*)softmax,
                mask, (float*)param_.d_score,
                head_num, batch, from_seq_len,
                to_seq_len, size_per_head, param_.dropout_rate, stream);
    } else {
        launch_backprop_masking_softmax<half>((half*)param_.grads, (half*)softmax,
                mask, (half*)param_.d_score,
                head_num, batch, from_seq_len,
                to_seq_len, size_per_head, param_.dropout_rate, stream);
    }

}

template void SoftmaxGrad<OperationType::FP32>::softmaxBack_kernelLauncher(
        cudaStream_t stream,
        const float* grads, // input
        const float* softmax, // input
        const bool* mask, // input
        const int batch,
        const int from_seq_len,
        const int to_seq_len,
        const int head_num,
        const int size_per_head
        );

template void SoftmaxGrad<OperationType::FP16>::softmaxBack_kernelLauncher(
        cudaStream_t stream,
        const Eigen::half* grads, // input
        const Eigen::half* softmax, // input
        const bool* mask, // input
        const int batch,
        const int from_seq_len,
        const int to_seq_len,
        const int head_num,
        const int size_per_head
        );

} //namespace softmaxbackprop


}//namespace multiheadattention
