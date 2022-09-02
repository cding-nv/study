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

#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/register_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/platform/stream_executor.h"

#include "src/common_op.h"
#include "src/grad_op.h"

namespace tensorflow {

using GPUDevice = Eigen::GpuDevice;
using namespace multiheadattention;

namespace functor {

template <typename T, typename DType>
struct LayernormGradOpFunctor<GPUDevice, T, DType> {
  static void  Compute(OpKernelContext* context,
                       layernormbackprop::LayernormGradParam<DType>& params,
                       int batch_size_,
                       int seq_len_,
                       int head_num_,
                       int hidden_units_,
                       int size_per_head_
                       ) {

      const cudaStream_t &stream = context->eigen_device<GPUDevice>().stream();
      auto* tf_stream = context->op_device_context()->stream();
      OP_REQUIRES(context, tf_stream, errors::Internal("No stream available."));
      params.stream = stream;
      params.tf_stream = tf_stream;
      multiheadattention::Allocator<AllocatorType::TF> allocator_(context, stream);
      layernormbackprop::LayernormGrad<TFTraits<T>::OpType> *attention = nullptr;

      attention = new layernormbackprop::LayernormGrad<TFTraits<T>::OpType>(allocator_, batch_size_,
              seq_len_, head_num_, hidden_units_, size_per_head_);

      OP_REQUIRES(context, attention != nullptr, errors::Internal("Allocation of attention failed."));

      attention->initialize(params);
      attention->backward();

      delete attention;
  }
};

} // end namespace functor

#ifdef GOOGLE_CUDA

#define REGISTER_KERNELS(T) \
    template struct functor::LayernormGradOpFunctor<GPUDevice, T, typename TFTraits<T>::DataType>;

#define REGISTER_GPU_KERNELS(T) \
    REGISTER_KERNELS(T);

TF_CALL_float(REGISTER_GPU_KERNELS);
//TF_CALL_half(REGISTER_GPU_KERNELS);

#undef REGISTER_GPU_KERNELS
#undef REGISTER_KERNELS

#endif

} // end namespace tensorflow

#endif

