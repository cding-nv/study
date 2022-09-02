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

#include "src/tf_op/non_zero_index_grad_op.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/register_types.h"


namespace tensorflow {

namespace {

using GPUDevice = Eigen::GpuDevice;
using CPUDevice = Eigen::ThreadPoolDevice;

template <typename T>
    class CommonOp : public OpKernel
{
    public:
        explicit CommonOp(OpKernelConstruction *context) : OpKernel(context) {
        };

        template<typename DataType_>
            void get_tensor(OpKernelContext *context, int tensor_id, const DataType_** tensor_ptr, int off_set = 0){
                *tensor_ptr = reinterpret_cast<const DataType_ *>(context->input(tensor_id).flat<T>().data()) + off_set;
                OP_REQUIRES(context, *tensor_ptr != nullptr, errors::InvalidArgument("tensor %d is null", tensor_id));
            }

        ~CommonOp() { /*cublasDestroy(cublas_handle_);*/ }
    private:
};

REGISTER_OP("NonZeroIndexGradGPU")
  .Input("data: T")
  .Output("output: int32")
  .Attr("T: {int32, float, half}")
  //.Attr("output_dim0: int")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
      auto batch = c->Dim(c->input(0), 0);
      auto len1 = c->Dim(c->input(0), 1);
      auto len2 = c->Dim(c->input(0), 2);
      auto len3 = c->Dim(c->input(0), 3);

      c->set_output(0, c->MakeShape({batch}));
      return Status::OK();
  });

template <typename Device, typename T>
class NonZeroIndexGradOp : public CommonOp<T> {
 public:
  explicit NonZeroIndexGradOp(OpKernelConstruction* context): CommonOp<T>(context) {
    //OP_REQUIRES_OK(context, context->GetAttr("num_segments", &num_segments_));
  }
  void Compute(OpKernelContext* context) override {

    auto data_shape  = context->input(0).shape();
    int batch = data_shape.dim_size(0);
    int len1  = data_shape.dim_size(1);
    int len2  = data_shape.dim_size(2);
    int len3  = data_shape.dim_size(3);

    int rank = (int)(context->input(0).dims());
    OP_REQUIRES(context, rank==4,
            errors::InvalidArgument("Invalid rank. The rank of from tensor should be 4\
                ([batch size, 1, 1, dimension]), but got ", rank));

    Tensor *output_tensor = nullptr;
    OP_REQUIRES_OK(
            context,
            context->allocate_output(0, {batch}, &output_tensor)
            );

    NonZeroIndexGradOpParams<DataType_> params;
    this->get_tensor(context, 0, &params.input_tensor);
    params.output_tensor = reinterpret_cast<int *>(output_tensor->flat<int>().data());
    params.batch = batch;
    params.total_len = len1 * len2 * len3;

    OP_REQUIRES_OK(context,
      functor::NonZeroIndexGradOpFunctor<Device, DataType_>::Compute(
        context->eigen_device<Device>(),
        params));
  }
 private:
  typedef TFTraits<T> traits_;
  typedef typename traits_::DataType DataType_;
};

#ifdef GOOGLE_CUDA

#define REGISTER_GPU(T)\
    REGISTER_KERNEL_BUILDER(\
        Name("NonZeroIndexGradGPU")\
        .Device(DEVICE_GPU)\
        .TypeConstraint<T>("T"),\
      NonZeroIndexGradOp<GPUDevice, T>)
//REGISTER_GPU(int);
REGISTER_GPU(float);
REGISTER_GPU(Eigen::half);

#undef REGISTER_GPU

#endif

} // end namespace

} // end namespace tensorflow
