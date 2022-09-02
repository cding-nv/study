/* 
 * Copyright (c) 2019, NVIDIA CORPORATION.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *      http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h>
#include <cuda_fp16.h>
#include "non_zero_index_grad_kernels.h"

template <typename T>
__global__ void compute_nonzero_index(const T* input_tensor,
        int batch, int total_len, int* output_tensor) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid < batch) {
        int src_id = tid * total_len;

        int index = tid + 1;
        if (input_tensor[src_id] == 0.0f &&
                input_tensor[src_id + 1] == 0.0f) {
            index = 0;
        }

        output_tensor[tid] = index;
    } // if
}

template <>
__global__ void compute_nonzero_index(const half* input_tensor,
        int batch, int total_len, int* output_tensor) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid < batch) {
        int src_id = tid * total_len;

        int index = tid + 1;
        if (__heq(input_tensor[src_id], __float2half(0.0f)) &&
                __heq(input_tensor[src_id + 1], __float2half(0.0f))) {
            index = 0;
        }

        output_tensor[tid] = index;
    }
}

template <typename T>
void non_zero_index_grad_launcher(NonZeroIndexGradOpParams<T>& params,
        cudaStream_t stream) {
    const int batch = params.batch;
    const int total_len = params.total_len;

    int block_size = 512;
    int grid_size = (batch + 511) / 512;

    compute_nonzero_index<T><<<grid_size, block_size, 0, stream>>>(
            params.input_tensor,
            batch, total_len,
            params.output_tensor);
}

template void non_zero_index_grad_launcher<int>(NonZeroIndexGradOpParams<int>& params,
        cudaStream_t stream);

template void non_zero_index_grad_launcher<half>(NonZeroIndexGradOpParams<half>& params,
        cudaStream_t stream);

template void non_zero_index_grad_launcher<float>(NonZeroIndexGradOpParams<float>& params,
        cudaStream_t stream);


