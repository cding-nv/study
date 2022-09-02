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

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "tensorflow/core/framework/register_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

template <typename T>
struct NonZeroIndexGradOpParams {
  int* output_tensor;   // index for non-zero rows
  const T* input_tensor;
  int batch;     // size: batch
  int total_len; // size: len0 * len1 * len2
};

template <typename T>
void non_zero_index_grad_launcher(NonZeroIndexGradOpParams<T>& params,
                                 cudaStream_t stream);

template <typename T> class TFTraits;

template <>
class TFTraits<float>
{
  public:
    typedef float DataType;
};

template <>
class TFTraits<Eigen::half>
{
  public:
    typedef __half DataType;
};

