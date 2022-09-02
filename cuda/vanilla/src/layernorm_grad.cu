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

namespace layernormbackprop{

template<OperationType OpType_>
void LayernormGrad<OpType_>::layernormBack_kernelLauncher(
      cudaStream_t stream,
      const DataType_* grads, // input
      const DataType_* x_data, // input
      const DataType_* vars, // input
      const DataType_* means, // input
      const DataType_* gamma, // input
      const int batch,
      const int seq_len,
      const int head_num,
      const int hidden_size,
      const int size_per_head)
{

    // These implementation is come from DeepSpeed with some modifications
    if (OpType_ == OperationType::FP32) {
        launch_layerNorm_backward<float>(
                (const float*)grads,
                (const float*)x_data,
                (const float*)vars,
                (const float*)means,
                (const float*)gamma,
                (float*)gamma_inter,
                (float*)betta_inter,
                (float*)param_.d_gamma,
                (float*)param_.d_betta,
                (float*)param_.d_x,
                head_num * batch * seq_len, // need to be confirmed
                size_per_head, // need to be confirmed
                batch, head_num, seq_len,
                param_.alpha,
                stream
                );
    } else {
        launch_layerNorm_backward<half>(
                (const half*)grads,
                (const half*)x_data,
                (const half*)vars,
                (const half*)means,
                (const half*)gamma,
                (half*)gamma_inter,
                (half*)betta_inter,
                (half*)param_.d_gamma,
                (half*)param_.d_betta,
                (half*)param_.d_x,
                head_num * batch * seq_len, // need to be confirmed
                size_per_head, // need to be confirmed
                batch, head_num, seq_len,
                param_.alpha,
                stream
                );
    }

    //OP_REQUIRES(param_.op_context, size_per_head % 32 == 0, errors::InvalidArgument("ERROR while calling ThenBlasGemmStridedBatched.\n"));

}

template void LayernormGrad<OperationType::FP32>::layernormBack_kernelLauncher(
        cudaStream_t stream,
        const float* grads, // input
        const float* x_data, // input
        const float* vars, // input
        const float* means, // input
        const float* gamma, // input
        const int batch,
        const int seq_len,
        const int head_num,
        const int hidden_size,
        const int size_per_head
        );

template void LayernormGrad<OperationType::FP16>::layernormBack_kernelLauncher(
        cudaStream_t stream,
        const Eigen::half* grads, // input
        const Eigen::half* x_data, // input
        const Eigen::half* vars, // input
        const Eigen::half* means, // input
        const Eigen::half* gamma, // input
        const int batch,
        const int seq_len,
        const int head_num,
        const int hidden_size,
        const int size_per_head
        );

} //namespace layernormbackprop


}//namespace multiheadattention
