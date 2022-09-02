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

namespace tensorflow {

namespace {

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("MultiHeadAttention")
    .Input("from_tensor: T")       // 0 [N, h, T_q,T_k]
    .Input("k_mask: bool")         // 1 [N, T_k] , removed q_mask [N, T_q]
    .Input("keep_mask: bool")      // 2 [B, N, F, T]
    .Output("output: T")           // 0 [B, N, F, T]
    .Output("softmax: T")          // 1 [B, N, F, T]
    //.Output("mask: bool")       // 2 [B, N, F, T], dropout mask
    .Attr("T: {float, half}")
    .Attr("batch: int >= 1")
    .Attr("head_num: int >= 1")
    .Attr("size_per_head: int >= 1")
    .Attr("dropout_rate: float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
            int head_num;
            c->GetAttr("head_num", &head_num);
            int batch;
            c->GetAttr("batch", &batch);

            auto T_q = c->Dim(c->input(0), 2);
            auto T_k = c->Dim(c->input(0), 3);

            // add parameters checking
            shape_inference::ShapeHandle unused_handle;
            TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &unused_handle));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &unused_handle));

            shape_inference::DimensionHandle unused_dhandle;
            TF_RETURN_IF_ERROR(c->Merge(c->Dim(c->input(0), 3), c->Dim(c->input(1), 1), &unused_dhandle));
            //printf("HEAD, FILE:%s, LINE:%d\n", __FILE__, __LINE__);

            c->set_output(0, c->MakeShape({batch, head_num, T_q, T_k}));
            c->set_output(1, c->MakeShape({batch, head_num, T_q, T_k}));
            //c->set_output(2, c->MakeShape({batch, head_num, T_q, T_k}));

            //printf("HEAD, FILE:%s, LINE:%d\n", __FILE__, __LINE__);
            return Status::OK();
    });

template <typename Device, typename T>
class MultiHeadAttentionOp : public CommonOp<T> {
 public:
  explicit MultiHeadAttentionOp(OpKernelConstruction* context): CommonOp<T>(context) {
      OP_REQUIRES_OK(context, context->GetAttr("batch", &batch_));
      OP_REQUIRES_OK(context, context->GetAttr("head_num", &head_num_));
      OP_REQUIRES_OK(context, context->GetAttr("size_per_head", &size_per_head_));
      OP_REQUIRES_OK(context, context->GetAttr("dropout_rate", &dropout_rate_));
      //printf("############# FILE: %s, LINE: %d\n", __FILE__, __LINE__);
  }

  void Compute(OpKernelContext* context) override {

      int batch_size_, from_seq_len_, to_seq_len_;

    //printf("############# FILE: %s, LINE: %d\n", __FILE__, __LINE__);

    batch_size_ = context->input(1).dim_size(0);    // N
    from_seq_len_ = context->input(0).dim_size(2);  // T_q
    to_seq_len_ = context->input(0).dim_size(3);    // T_k

    // print the shape of all the inputs
    //std::cout << "from_tensor: ["
    //    << context->input(0).dim_size(0) <<", "
    //    << context->input(0).dim_size(1) <<", "
    //    << context->input(0).dim_size(2) << "]"
    //    << std::endl;

    //
    OP_REQUIRES(context, dropout_rate_ >=0.0f && dropout_rate_ < 1.0f,
                errors::InvalidArgument("Invalid dropout rate, dropout rate should be within [0, 1)."));

    OP_REQUIRES(context, batch_ == batch_size_,
                errors::InvalidArgument("Invalid batch size, input batch and retrieved batch are not equal: ", batch_size_, " vs ", batch_));

    OP_REQUIRES(context, from_seq_len_ > 0 && to_seq_len_ > 0,
                errors::InvalidArgument("Dimension for each input should be > 0"));

    int rank = (int)context->input(0).dims();
    OP_REQUIRES(context, rank==4,
                errors::InvalidArgument("Invalid rank. The rank of from tensor should be 3\
                                        ([batch size,  head_num, from sequence length, to_seq_len]), but got ", rank));

    rank = (int)(context->input(1).dims());
    OP_REQUIRES(context, rank==2,
                errors::InvalidArgument("Invalid rank. The rank of K tensor should be 3\
                                        ([batch size, to sequence length]), but got ", rank));

    OP_REQUIRES(context, context->num_inputs() == 3, errors::InvalidArgument("Less input arguments"));


    Tensor * output = nullptr;
    OP_REQUIRES_OK(
            context,
            context->allocate_output(0, {batch_size_,  head_num_, from_seq_len_, to_seq_len_}, &output));
    Tensor * out1 = nullptr;
    OP_REQUIRES_OK(
            context,
            context->allocate_output(1, {batch_size_, head_num_, from_seq_len_, to_seq_len_}, &out1));
    //Tensor * out2 = nullptr;
    //OP_REQUIRES_OK(
            //context,
            //context->allocate_output(2, {batch_size_, head_num_, from_seq_len_, to_seq_len_}, &out2));

    //printf("############# FILE: %s, LINE: %d\n", __FILE__, __LINE__);
    if (output->NumElements() == 0)
        return;
    if (out1->NumElements() == 0)
        return;

    cuda::MultiHeadInitParam<DataType_> param;
    this->get_tensor(context, 0, &param.attention_scores);
    param.k_mask = reinterpret_cast<const bool *>(context->input(1).flat<bool>().data());
    param.mask = reinterpret_cast<const bool *>(context->input(2).flat<bool>().data());
    param.dropout_rate = dropout_rate_;

    param.op_context = context;

    param.output = reinterpret_cast<DataType_ *>(output->flat<T>().data());
    param.softmax = reinterpret_cast<DataType_ *>(out1->flat<T>().data());
    //param.mask = reinterpret_cast<bool *>(out2->flat<bool>().data());

    //printf("############# FILE: %s, LINE: %d\n", __FILE__, __LINE__);
    functor::MultiHeadAttentionOpFunctor<Device, T, DataType_>::Compute(
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
  int batch_, head_num_, size_per_head_;
  float dropout_rate_;
  typedef TFTraits<T> traits_;
  typedef typename traits_::DataType DataType_;
};

#ifdef GOOGLE_CUDA

#define REGISTER_GPU(T)                 \
    REGISTER_KERNEL_BUILDER(            \
        Name("MultiHeadAttention")      \
        .Device(DEVICE_GPU)             \
        .TypeConstraint<T>("T"),        \
      MultiHeadAttentionOp<GPUDevice, T>);

REGISTER_GPU(float);
//REGISTER_GPU(double);
//REGISTER_GPU(Eigen::half);

#undef REGISTER_GPU

#endif

} // end namespace

} // end namespace tensorflow
