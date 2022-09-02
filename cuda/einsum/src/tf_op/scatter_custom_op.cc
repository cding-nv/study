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

#include "src/tf_op/scatter_custom_op.h"

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

REGISTER_OP("ScatterCustomGPU")
  .Input("position: int32")
  .Input("data: T")
  .Output("output: T")
  .Attr("T: {int32, float, half}")
  .Attr("shape1: int >= 1")
  .Attr("shape2: int >= 1")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
      int shape1;
      int shape2;
      //auto batch = c->Dim(c->input(0), 0);
      //auto len1 = c->Dim(c->input(0), 1);
      //auto len2 = c->Dim(c->input(0), 2);
      //auto len3 = c->Dim(c->input(0), 3);
      c->GetAttr("shape1", &shape1);
      c->GetAttr("shape2", &shape2);
      //const Tensor* shape = c->input_tensor(2);
      
      //std::cout << "###shape " << c->Value(shape) << std::endl;
      c->set_output(0, c->MakeShape({shape1, shape2}));
      //c->set_output(0, c->input(2));
      std::cout << "## shape1=" << shape1 << " shape2="<< shape2 << std::endl;
      return Status::OK();
  });

template <typename Device, typename T>
class ScatterCustomOp : public CommonOp<T> {
 public:
  explicit ScatterCustomOp(OpKernelConstruction* context): CommonOp<T>(context) {
    //OP_REQUIRES_OK(context, context->GetAttr("num_segments", &num_segments_));
    OP_REQUIRES_OK(context, context->GetAttr("shape1", &shape1_));
    OP_REQUIRES_OK(context, context->GetAttr("shape2", &shape2_));
  }
  void Compute(OpKernelContext* context) override {
    ScatterCustomOpParams<DataType_> params;
    this->get_tensor(context, 1, &params.input_tensor);
    //std::cout << "#### shape " << params.input_tensor[0] << std::endl;
    //this->get_tensor(context, 0, &params.index_tensor);
    params.index_tensor = reinterpret_cast<const int *>(context->input(0).flat<int>().data());

    auto data_shape  = context->input(0);
    int index_shape = data_shape.dim_size(0);
    std::cout << "## index shape " << index_shape << std::endl;

    int rank = (int)(context->input(1).dims());
    OP_REQUIRES(context, rank==4,
            errors::InvalidArgument("Invalid rank. The rank of from tensor should be 4\
                ([batch size, 1, 1, dimension]), but got ", rank));

    Tensor *output_tensor = nullptr;
    OP_REQUIRES_OK(
            context,
            context->allocate_output(0, {shape1_, 1, 1, shape2_}, &output_tensor)
            );

    //params.output_tensor = reinterpret_cast<float *>(output_tensor->flat<int>().data());
    params.output_tensor = reinterpret_cast<DataType_ *>(output_tensor->flat<T>().data());
    params.output_shape0 = shape1_;
    params.output_shape3 = shape2_;
    params.index_shape = index_shape;

    OP_REQUIRES_OK(context,
      functor::ScatterCustomOpFunctor<Device, DataType_>::Compute(
        context->eigen_device<Device>(),
        params));
  }
 public:
  int shape1_;
  int shape2_;
  typedef TFTraits<T> traits_;
  typedef typename traits_::DataType DataType_;
};

#ifdef GOOGLE_CUDA

#define REGISTER_GPU(T)\
    REGISTER_KERNEL_BUILDER(\
        Name("ScatterCustomGPU")\
        .Device(DEVICE_GPU)\
        .TypeConstraint<T>("T"),\
      ScatterCustomOp<GPUDevice, T>)
//REGISTER_GPU(int);
REGISTER_GPU(float);
REGISTER_GPU(Eigen::half);

#undef REGISTER_GPU

#endif

} // end namespace

} // end namespace tensorflow
