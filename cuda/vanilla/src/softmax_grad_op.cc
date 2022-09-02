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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "src/common_op.h"
#include "src/grad_op.h"

namespace tensorflow {

namespace {

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("SoftmaxGrad")
    .Input("grads: T")                 // 0 [B,N,F,T]
    .Input("softmax: T")               // 1 [B,N,F,T]
    .Input("dropout_mask: bool")       // 2 [B,N,F,T], dropout mask
    .Output("d_score: T")              // 0 [B, N, F, T]
    //.Output("tmp: T")                // 1 TBD
    .Attr("T: {float, half}")
    .Attr("head_num: int >= 1")
    .Attr("batch: int >= 1")
    .Attr("size_per_head: int >= 1")
    .Attr("dropout_rate: float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
            int head_num;
            c->GetAttr("head_num", &head_num);
            int batch;
            c->GetAttr("batch", &batch);
            int size_per_head;
            c->GetAttr("size_per_head", &size_per_head);

            auto T_q = c->Dim(c->input(0), 2);
            auto T_k = c->Dim(c->input(0), 3);

            // add parameters checking
            shape_inference::ShapeHandle unused_handle;
            TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &unused_handle));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &unused_handle));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 4, &unused_handle));

            shape_inference::DimensionHandle unused_dhandle;
            TF_RETURN_IF_ERROR(c->Merge(c->Dim(c->input(0), 0), c->Dim(c->input(1), 0), &unused_dhandle));
            TF_RETURN_IF_ERROR(c->Merge(c->Dim(c->input(0), 1), c->Dim(c->input(1), 1), &unused_dhandle));
            TF_RETURN_IF_ERROR(c->Merge(c->Dim(c->input(0), 2), c->Dim(c->input(1), 2), &unused_dhandle));
            TF_RETURN_IF_ERROR(c->Merge(c->Dim(c->input(0), 3), c->Dim(c->input(1), 3), &unused_dhandle));

            TF_RETURN_IF_ERROR(c->Merge(c->Dim(c->input(2), 0), c->Dim(c->input(0), 0), &unused_dhandle));
            TF_RETURN_IF_ERROR(c->Merge(c->Dim(c->input(2), 1), c->Dim(c->input(0), 1), &unused_dhandle));
            TF_RETURN_IF_ERROR(c->Merge(c->Dim(c->input(2), 2), c->Dim(c->input(0), 2), &unused_dhandle));
            TF_RETURN_IF_ERROR(c->Merge(c->Dim(c->input(2), 3), c->Dim(c->input(0), 3), &unused_dhandle));

            //if (hidden_units > 1024) {
            //    return errors::InvalidArgument (
            //            "Hidden units should not be larger than 1024.");
            //}


            c->set_output(0, c->MakeShape({batch, head_num, T_q, T_k}));
            //c->set_output(1, c->MakeShape({head_num*batch, T_k, size_per_head}));

            return Status::OK();
    });

template <typename Device, typename T>
class SoftmaxGradOp : public CommonOp<T> {
 public:
  explicit SoftmaxGradOp(OpKernelConstruction* context): CommonOp<T>(context) {
      OP_REQUIRES_OK(context, context->GetAttr("head_num", &head_num_));
      OP_REQUIRES_OK(context, context->GetAttr("batch", &batch_size_));
      OP_REQUIRES_OK(context, context->GetAttr("size_per_head", &size_per_head_));
      OP_REQUIRES_OK(context, context->GetAttr("dropout_rate", &dropout_rate_));
  }

  void Compute(OpKernelContext* context) override {

      int from_seq_len_, to_seq_len_;

      batch_size_ = context->input(0).dim_size(0);    // N
      from_seq_len_ = context->input(0).dim_size(2);  // T_q
      to_seq_len_ = context->input(0).dim_size(3);    // T_k

    // print the shape of all the inputs
    //std::cout << "from_tensor: ["
    //    << context->input(0).dim_size(0) <<", "
    //    << context->input(0).dim_size(1) <<", "
    //    << context->input(0).dim_size(2) << "]"
    //    << std::endl;

    OP_REQUIRES(context, dropout_rate_ >= 0.0f && dropout_rate_ < 1.0f,
                errors::InvalidArgument("Invaid dropout rate, dropout rate should be within [0,1)."));

    OP_REQUIRES(context, from_seq_len_ > 0 && to_seq_len_ > 0 && batch_size_ > 0,
                errors::InvalidArgument("Dimension for each input should be > 0"));

    int rank = (int)context->input(0).dims();
    OP_REQUIRES(context, rank==4,
                errors::InvalidArgument("Invalid rank. The rank of grads tensor should be 3\
                                        ([batch size, head_num, from sequence length, to seq len]), but got ", rank));

    rank = (int)(context->input(1).dims());
    OP_REQUIRES(context, rank==4,
                errors::InvalidArgument("Invalid rank. The rank of K tensor should be 3\
                                        ([batch size, head_num, from_seq_len, to_seq_len]), but got ", rank));
    //M <= N || N == 0
    //OP_REQUIRES(context, batch_size_kv_ <= batch_size_ || batch_size_ == 0,
                //errors::InvalidArgument("Invalid shape, the requirement: batch_size_kv_ <= batch_size_ || batch_size_ == 0: ", batch_size_kv_, " vs ", batch_size_));

//    rank = (int)(context->input(2).dims());
//    OP_REQUIRES(context, rank==3,
//                errors::InvalidArgument("Invalid rank. The rank of V tensor should be 3\
//                                        ([batch size, sequence length, hidden dimension]), but got ", rank));
//    OP_REQUIRES(context, (batch_size_kv_ == context->input(2).dim_size(0)) && (to_seq_len_ == context->input(2).dim_size(1)),
//                errors::InvalidArgument("Invalid shape: The first two dimension of K and V should be the same. ",
//                    context->input(1).shape().DebugString(), " vs ", context->input(2).shape().DebugString()));
//
//
    OP_REQUIRES(context, context->num_inputs() == 3, errors::InvalidArgument("Less input arguments"));

    Tensor * d_score_ = nullptr;
    OP_REQUIRES_OK(
            context,
            context->allocate_output(0, {batch_size_, head_num_, from_seq_len_, to_seq_len_}, &d_score_));

//    Tensor * d_tmp_ = nullptr;
//    OP_REQUIRES_OK(
//            context,
//            context->allocate_output(3, {head_num_ * batch_size_, from_seq_len_, size_per_head_out_}, &d_tmp_));

    if (d_score_->NumElements() == 0)
        return;

    softmaxbackprop::SoftmaxGradParam<DataType_> param;
    param.op_context = context;
    //param.cublas_handle = this->get_cublas_handler();
    //check_cuda_error(cublasSetStream(param.cublas_handle, param.stream));
    this->get_tensor(context, 0, &param.grads);
    this->get_tensor(context, 1, &param.softmax);
    param.mask = reinterpret_cast<const bool *>(context->input(2).flat<bool>().data());

    param.d_score = reinterpret_cast<DataType_ *>(d_score_->flat<T>().data());
    param.dropout_rate = dropout_rate_;
    //param.d_tmp = reinterpret_cast<DataType_ *>(d_tmp_->flat<T>().data());

    functor::SoftmaxGradOpFunctor<Device, T, DataType_>::Compute(
            context,
            param,
            batch_size_,
            from_seq_len_,
            to_seq_len_,
            head_num_,
            size_per_head_
            );
  }
 private:
  float dropout_rate_;
  int head_num_, batch_size_, size_per_head_;
  typedef TFTraits<T> traits_;
  typedef typename traits_::DataType DataType_;
};

#ifdef GOOGLE_CUDA

#define REGISTER_GPU(T)                 \
    REGISTER_KERNEL_BUILDER(            \
        Name("SoftmaxGrad")      \
        .Device(DEVICE_GPU)             \
        .TypeConstraint<T>("T"),        \
      SoftmaxGradOp<GPUDevice, T>);

REGISTER_GPU(float);
//REGISTER_GPU(double);
//REGISTER_GPU(Eigen::half);

#undef REGISTER_GPU

#endif

} // end namespace

} // end namespace tensorflow
