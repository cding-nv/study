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

namespace densebackprop{

template<OperationType OpType_>
void DenseGrad<OpType_>::denseBack_kernelLauncher(
      cudaStream_t stream,
      const DataType_* q_grads, // input
      const DataType_* k_grads, // input
      const DataType_* v_grads, // input
      const DataType_* query, // input
      const DataType_* key, // input
      const DataType_* value, // input
      const DataType_* q_kernel, // input
      const DataType_* k_kernel, // input
      const DataType_* v_kernel, // input
      const DataType_* value_layer, // input
      const int batch,
      const int from_seq_len,
      const int to_seq_len,
      const int head_num,
      const int hidden_size,
      const int size_per_head,
      const int size_per_head_out,
      const int hs_q,
      const int hs_k,
      const int hs_v
      )
{

    int hidden_size_out = size_per_head_out * head_num;

    float alpha = 1.0f, beta = 0.0f;
    bool blas_launch_status = false;

    // Computation of Query
    // [N, T_q, C]
    auto b_ptr = ToDeviceMemory(q_grads);
    // [C_q, C]
    auto a_ptr = ToDeviceMemory(q_kernel);
    // [N, T_q, C_q]
    auto c_ptr = ToDeviceMemory(param_.dq);

    blas_launch_status = param_.tf_stream
        ->ThenBlasGemm(
                blas::Transpose::kTranspose,
                blas::Transpose::kNoTranspose,
                (uint64)hs_q, (uint64)(batch * from_seq_len), (uint64)hidden_size,
                alpha,
                a_ptr, hidden_size,
                b_ptr, hidden_size,
                beta,
                &c_ptr, hs_q
                ).ok();
    OP_REQUIRES(param_.op_context, blas_launch_status, errors::InvalidArgument("ERROR while calling ThenBlasGemm.\n"));

    // [N, T_q, C_q]
    b_ptr = ToDeviceMemory(query);
    // [N, T_q, C]
    a_ptr = ToDeviceMemory(q_grads);
    // [N, C_q, C]
    c_ptr = ToDeviceMemory(dwq_inter);

    blas_launch_status = param_.tf_stream
        -> ThenBlasGemmStridedBatched(
            blas::Transpose::kNoTranspose, blas::Transpose::kTranspose,
            (uint64)(hidden_size), (uint64)hs_q, (uint64)from_seq_len,
            alpha,
            a_ptr, hidden_size, (uint64)(hidden_size * from_seq_len),
            b_ptr, hs_q, (uint64)(hs_q * from_seq_len),
            beta,
            &c_ptr, hidden_size, (uint64)(hidden_size * hs_q),
            batch
                ).ok();
    OP_REQUIRES(param_.op_context, blas_launch_status, errors::InvalidArgument("ERROR while calling ThenBlasGemmStridedBatched.\n"));

    // Computation of Key
    // [N, T_k, C]
    b_ptr = ToDeviceMemory(k_grads);
    // [C_k, C]
    a_ptr = ToDeviceMemory(k_kernel);
    // [N, T_k, C_k]
    c_ptr = ToDeviceMemory(param_.dk);

    blas_launch_status = param_.tf_stream
        ->ThenBlasGemm(
                blas::Transpose::kTranspose,
                blas::Transpose::kNoTranspose,
                (uint64)hs_k, (uint64)(batch * to_seq_len), (uint64)hidden_size,
                alpha,
                a_ptr, hidden_size,
                b_ptr, hidden_size,
                beta,
                &c_ptr, hs_k
                ).ok();
    OP_REQUIRES(param_.op_context, blas_launch_status, errors::InvalidArgument("ERROR while calling ThenBlasGemm.\n"));

    // [N, T_k, C_k]
    b_ptr = ToDeviceMemory(key);
    // [N, T_k, C]
    a_ptr = ToDeviceMemory(k_grads);
    // [N, C_k, C]
    c_ptr = ToDeviceMemory(dwk_inter);

    blas_launch_status = param_.tf_stream
        -> ThenBlasGemmStridedBatched(
            blas::Transpose::kNoTranspose, blas::Transpose::kTranspose,
            (uint64)(hidden_size), (uint64)hs_k, (uint64)to_seq_len,
            alpha,
            a_ptr, hidden_size, (uint64)(hidden_size * to_seq_len),
            b_ptr, hs_k, (uint64)(hs_k * to_seq_len),
            beta,
            &c_ptr, hidden_size, (uint64)(hidden_size * hs_k),
            batch
                ).ok();
    OP_REQUIRES(param_.op_context, blas_launch_status, errors::InvalidArgument("ERROR while calling ThenBlasGemmStridedBatched.\n"));

    // Add leaky relu backprop
    if (OpType_ == OperationType::FP32) {
        //cudaMemcpy(param_.dtmp, v_grads, sizeof(float)*batch*to_seq_len*hidden_size_out, cudaMemcpyDeviceToDevice);
        launch_transpose1203_lreluback<float>((const float*)v_grads, (float*)v_grads_inter, (const float*)value_layer, param_.alpha,
                batch, to_seq_len, head_num, size_per_head_out, stream);

    } else {
        launch_transpose1203_lreluback<half>((const half*)v_grads, (half*)v_grads_inter, (const half*)value_layer, param_.alpha,
                batch, to_seq_len, head_num, size_per_head_out, stream);
    }

    // Computation of Value
    // [N, T_k, O]
    b_ptr = ToDeviceMemory(v_grads_inter);
    // [C_v, O]
    a_ptr = ToDeviceMemory(v_kernel);
    // [N, T_k, C_v]
    c_ptr = ToDeviceMemory(param_.dv);

    blas_launch_status = param_.tf_stream
        ->ThenBlasGemm(
                blas::Transpose::kTranspose,
                blas::Transpose::kNoTranspose,
                (uint64)hs_v, (uint64)(batch * to_seq_len), (uint64)hidden_size_out,
                alpha,
                a_ptr, hidden_size_out,
                b_ptr, hidden_size_out,
                beta,
                &c_ptr, hs_v
                ).ok();
    OP_REQUIRES(param_.op_context, blas_launch_status, errors::InvalidArgument("ERROR while calling ThenBlasGemm.\n"));

    // [N, T_k, C_v]
    b_ptr = ToDeviceMemory(value);
    // [N, T_k, O]
    a_ptr = ToDeviceMemory(v_grads_inter);
    // [N, C_v, O]
    c_ptr = ToDeviceMemory(dwv_inter);

    blas_launch_status = param_.tf_stream
        -> ThenBlasGemmStridedBatched(
            blas::Transpose::kNoTranspose, blas::Transpose::kTranspose,
            (uint64)(hidden_size_out), (uint64)hs_v, (uint64)to_seq_len,
            alpha,
            a_ptr, hidden_size_out, (uint64)(hidden_size_out * to_seq_len),
            b_ptr, hs_v, (uint64)(hs_v * to_seq_len),
            beta,
            &c_ptr, hidden_size_out, (uint64)(hidden_size_out * hs_v),
            batch
                ).ok();
    OP_REQUIRES(param_.op_context, blas_launch_status, errors::InvalidArgument("ERROR while calling ThenBlasGemmStridedBatched.\n"));

    if (OpType_ == OperationType::FP32) {

        //launch_sum_reduce_w<float>( (const float*)dwq_inter,
        //        (const float*)dwk_inter,
        //        (const float*)dwv_inter,
        //        (float *)param_.dwq,
        //        (float *)param_.dwk,
        //        (float *)param_.dwv,
        //        batch,
        //        hidden_size,
        //        hs_q,
        //        hs_k,
        //        hs_v,
        //        stream);

        launch_sum_reduce_w<float>( (const float*)dwq_inter,
                (float *)param_.dwq,
                batch,
                hidden_size,
                hs_q,
                stream);

        launch_sum_reduce_w<float>( (const float*)dwk_inter,
                (float *)param_.dwk,
                batch,
                hidden_size,
                hs_k,
                stream);

        launch_sum_reduce_w<float>( (const float*)dwv_inter,
                (float *)param_.dwv,
                batch,
                hidden_size_out,
                hs_v,
                stream);

        launch_column_sum_reduce<float>(
                (const float *)q_grads,
                (float *)param_.dbq,
                batch * from_seq_len,
                hidden_size,
                stream);

        launch_column_sum_reduce<float>(
                (const float *)k_grads,
                (float *)param_.dbk,
                batch * to_seq_len,
                hidden_size,
                stream);

        launch_column_sum_reduce<float>(
                (const float *)v_grads_inter,
                (float *)param_.dbv,
                batch * to_seq_len,
                hidden_size_out,
                stream);
    } else {
        launch_sum_reduce_w<half>( (const half*)dwq_inter,
                (half *)param_.dwq,
                batch,
                hidden_size,
                hs_q,
                stream);

        launch_sum_reduce_w<half>( (const half*)dwk_inter,
                (half *)param_.dwk,
                batch,
                hidden_size,
                hs_k,
                stream);

        launch_sum_reduce_w<half>( (const half*)dwv_inter,
                (half *)param_.dwv,
                batch,
                hidden_size_out,
                hs_v,
                stream);

        launch_column_sum_reduce<half>(
                (const half *)q_grads,
                (half *)param_.dbq,
                batch * from_seq_len,
                hidden_size,
                stream);

        launch_column_sum_reduce<half>(
                (const half *)k_grads,
                (half *)param_.dbk,
                batch * to_seq_len,
                hidden_size,
                stream);

        launch_column_sum_reduce<half>(
                (const half *)v_grads_inter,
                (half *)param_.dbv,
                batch * to_seq_len,
                hidden_size_out,
                stream);
    }
    //OP_REQUIRES(param_.op_context, size_per_head % 32 == 0, errors::InvalidArgument("ERROR while calling ThenBlasGemmStridedBatched.\n"));
}

template void DenseGrad<OperationType::FP32>::denseBack_kernelLauncher(
        cudaStream_t stream,
        const float* q_grads, // input
        const float* k_grads, // input
        const float* v_grads, // input
        const float* query, // input
        const float* key, // input
        const float* value, // input
        const float* q_kernel, // input
        const float* k_kernel, // input
        const float* v_kernel, // input
        const float* value_layer, // input
        const int batch,
        const int from_seq_len,
        const int to_seq_len,
        const int head_num,
        const int hidden_size,
        const int size_per_head,
        const int size_per_head_out,
        const int hs_q,
        const int hs_k,
        const int hs_v
);

template void DenseGrad<OperationType::FP16>::denseBack_kernelLauncher(
        cudaStream_t stream,
        const Eigen::half* q_grads, // input
        const Eigen::half* k_grads, // input
        const Eigen::half* v_grads, // input
        const Eigen::half* query, // input
        const Eigen::half* key, // input
        const Eigen::half* value, // input
        const Eigen::half* q_kernel, // input
        const Eigen::half* k_kernel, // input
        const Eigen::half* v_kernel, // input
        const Eigen::half* value_layer, // input
        const int batch,
        const int from_seq_len,
        const int to_seq_len,
        const int head_num,
        const int hidden_size,
        const int size_per_head,
        const int size_per_head_out,
        const int hs_q,
        const int hs_k,
        const int hs_v
        );

} //namespace densebackprop

}//namespace multiheadattention
