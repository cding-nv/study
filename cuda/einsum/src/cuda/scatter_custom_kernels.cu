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
#include "scatter_custom_kernels.h"

// Device code
//template <>
__global__ void ScatterNDKernel(const int *indices,
    int index_rank,
    //const int *data_shape,
    int data_shape,  // todo: output dims[0]
    const float* updates,
    int vec_size,
    float tar_num,
    float* output
) 
{
    int tar_idx = blockIdx.x;

    // calculate the index part idx based on indice input
    int indices_base =  tar_idx * index_rank;
    int index_idx = 0;
    for (int i=0; i<index_rank; ++i){
	    //printf("data_shape[%d]: %d\n", i, data_shape[i]);
        index_idx = index_idx * data_shape + (int)indices[indices_base+i];
    }

    // calculate the source idx of params
    int data_idx_base = index_idx * vec_size;

    // calculate the target idx of updates
    int updates_idx_base = tar_idx * vec_size;

    int vec_idx, data_idx, updates_idx;
    for (int i=0; i<tar_num; ++i){
        vec_idx = threadIdx.x + blockDim.x * i;
        if (vec_idx>=vec_size){
            break;
        }
        data_idx = data_idx_base + vec_idx;
        updates_idx = updates_idx_base + vec_idx;
        float* data = output+data_idx;
        //if (*data > 0.0001)
            //printf("data =%f blockDim.x=%d threadIdx.x=%d index_idx=%d indices[base]=%d data_idx=%d updates_idx=%d\n", *data, blockDim.x, threadIdx.x, index_idx, indices[indices_base], data_idx, updates_idx);
	    atomicExch(output+data_idx, updates[updates_idx]);
    }
}

//template <>
__global__ void ScatterNDKernel(const int *indices,
    int index_rank,
    //const int *data_shape,
    int data_shape,  // todo: output dims[0]
    const int* updates,
    int vec_size,
    float tar_num,
    int* output
) 
{
    int tar_idx = blockIdx.x;

    // calculate the index part idx based on indice input
    int indices_base =  tar_idx * index_rank;
    int index_idx = 0;
    for (int i=0; i<index_rank; ++i){
	    //printf("data_shape[%d]: %d\n", i, data_shape[i]);
        index_idx = index_idx * data_shape + indices[indices_base+i];
    }

    // calculate the source idx of params
    int data_idx_base = index_idx * vec_size;

    // calculate the target idx of updates
    int updates_idx_base = tar_idx * vec_size;

    int vec_idx, data_idx, updates_idx;
    for (int i=0; i<tar_num; ++i){
        vec_idx = threadIdx.x + blockDim.x * i;
        if (vec_idx>=vec_size){
            break;
        }
        data_idx = data_idx_base + vec_idx;
        updates_idx = updates_idx_base + vec_idx;

	    atomicExch(output+data_idx, updates[updates_idx]);
    }
}

//template <>
__global__ void ScatterNDKernel(const int *indices,
    int index_rank,
    //const int *data_shape,
    int data_shape,  // todo: output dims[0]
    const half* updates,
    int vec_size,
    float tar_num,
    half* output
) 
{
    int tar_idx = blockIdx.x;

    // calculate the index part idx based on indice input
    int indices_base =  tar_idx * index_rank;
    int index_idx = 0;
    for (int i=0; i<index_rank; ++i){
	    //printf("data_shape[%d]: %d\n", i, data_shape[i]);
        index_idx = index_idx * data_shape + indices[indices_base+i];
    }

    // calculate the source idx of params
    int data_idx_base = index_idx * vec_size;

    // calculate the target idx of updates
    int updates_idx_base = tar_idx * vec_size;

    int vec_idx, data_idx, updates_idx;
    for (int i=0; i<tar_num; ++i){
        vec_idx = threadIdx.x + blockDim.x * i;
        if (vec_idx>=vec_size){
            break;
        }
        data_idx = data_idx_base + vec_idx;
        updates_idx = updates_idx_base + vec_idx;

	    //atomicExch(output+data_idx, updates[updates_idx]);
    }
}

template <typename T>
void scatter_custom_launcher(ScatterCustomOpParams<T>& params,
        cudaStream_t stream) {
    // const int batch = params.batch;
    // const int total_len = params.total_len;

    // int block_size = 512;
    // int grid_size = (batch + 511) / 512;

    // nonzero_index<T><<<grid_size, block_size, 0, stream>>>(
    //         params.input_tensor,
    //         batch, total_len,
    //         params.output_tensor);

    dim3 dimBlock;
    dim3 dimGrid;
    int index_rank = 1; // index.dims[nbDims - 1]
    int tar_size = params.index_shape; // index number
    int vec_size = 1 * params.output_shape3; // output.dims[index_rank] * ... * output.dims[nbDims - index_rank]

    //int data_shape = 10240; // todo: output dims[0]

    dimBlock.x = vec_size >= 1024 ? 1024 : vec_size;
    dimGrid.x = tar_size;
    printf("dimGrid.x=%d, dimBlock.x=%d\n", dimGrid.x, dimBlock.x);
    printf("params.output_shape0 = %d\n", params.output_shape0);
    printf("vec_size = %d\n", vec_size);
    printf("### calling ScatterNDKernel\n");

    // tf context->allocate_output() does not init 0 
    cudaMemsetAsync((void*)params.output_tensor, 0, params.output_shape0 * params.output_shape3 * sizeof(T), stream);

    // invoke kernel
    ScatterNDKernel<<<dimGrid, dimBlock, 0, stream>>>(params.index_tensor,  // index
                                                index_rank,
                                                params.output_shape0,
                                                params.input_tensor,     // data update
                                                vec_size,
                                                ceilf(vec_size/float(dimBlock.x)),
                                                params.output_tensor);
    printf("### end of ScatterNDKernel\n");
}

template void scatter_custom_launcher<int>(ScatterCustomOpParams<int>& params,
        cudaStream_t stream);

template void scatter_custom_launcher<half>(ScatterCustomOpParams<half>& params,
        cudaStream_t stream);

template void scatter_custom_launcher<float>(ScatterCustomOpParams<float>& params,
        cudaStream_t stream);


