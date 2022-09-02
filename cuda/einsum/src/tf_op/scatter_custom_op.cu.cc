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

#include <cuda.h>
#include "src/tf_op/scatter_custom_op.h"
#include "tensorflow/core/framework/op.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

using GPUDevice = Eigen::GpuDevice;

namespace functor {

template <typename T>
struct ScatterCustomOpFunctor<GPUDevice, T> {
  static Status Compute(const GPUDevice& d,
                        ScatterCustomOpParams<T>& params) {
      const cudaStream_t &stream = d.stream();
      scatter_custom_launcher<T>(params, stream);
      return Status::OK();
  }
};

} // end namespace functor

#ifdef GOOGLE_CUDA

#define REGISTER_KERNELS(T) \
    template struct functor::ScatterCustomOpFunctor<GPUDevice, T>;

#define REGISTER_GPU_KERNELS(T) \
    REGISTER_KERNELS(typename TFTraits<T>::DataType);

TF_CALL_float(REGISTER_GPU_KERNELS);
TF_CALL_half(REGISTER_GPU_KERNELS);

#undef REGISTER_GPU_KERNELS
#undef REGISTER_KERNELS

#endif


} // end namespace tensorflow

#endif

