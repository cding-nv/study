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

REGISTER_OP("LayernormGrad")
    .Input("grads: T")                 // 0 [h*N,T,C/h]
    .Input("x_data: T")                // 1 [h*N,T,C/h]
    .Input("vars: T")                  // 2 [h*N,T,1]
    .Input("means: T")                 // 3 [h*N,T,1]
    .Input("gamma: T")                 // 4 [C/h]
    .Output("d_gamma: T")              // 0 [C/h]
    .Output("d_betta: T")              // 1 [C/h]
    .Output("d_x: T")                  // 2 [h*N,T,C/h]
    //.Output("tmp: T")                  // 3 TBD
    .Attr("T: {float, half}")
    .Attr("head_num: int >= 1")
    .Attr("hidden_units: int >= 1")
    .Attr("batch: int >= 1")
    .Attr("size_per_head: int >= 1")
    .Attr("alpha: float") // alpha in leaky_relu
    .SetShapeFn([](shape_inference::InferenceContext* c) {
            int head_num;
            c->GetAttr("head_num", &head_num);
            int hidden_units;
            c->GetAttr("hidden_units", &hidden_units);
            int batch;
            c->GetAttr("batch", &batch);
            int size_per_head;
            c->GetAttr("size_per_head", &size_per_head);

            //auto batch = c->Dim(c->input(0), 0);
            auto seq_len = c->Dim(c->input(0), 1);
            //auto size_per_head = c->Dim(c->input(2), 2);

            // add parameters checking
            //shape_inference::ShapeHandle unused_handle;
            //TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &unused_handle));
            //TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &unused_handle));
            //TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 3, &unused_handle));
            //TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &unused_handle));
            //TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 1, &unused_handle));
            //TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 2, &unused_handle));
            //TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 1, &unused_handle));
            //TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 2, &unused_handle));
            //TF_RETURN_IF_ERROR(c->WithRank(c->input(8), 1, &unused_handle));
            //TF_RETURN_IF_ERROR(c->WithRank(c->input(9), 2, &unused_handle));
            //TF_RETURN_IF_ERROR(c->WithRank(c->input(10), 2, &unused_handle));
            //TF_RETURN_IF_ERROR(c->WithRank(c->input(11), 1, &unused_handle));
            //TF_RETURN_IF_ERROR(c->WithRank(c->input(12), 1, &unused_handle));
            //TF_RETURN_IF_ERROR(c->WithRank(c->input(13), 1, &unused_handle));
            //TF_RETURN_IF_ERROR(c->WithRank(c->input(14), 1, &unused_handle));

            //shape_inference::DimensionHandle unused_dhandle;
            //TF_RETURN_IF_ERROR(c->Merge(c->Dim(c->input(0), 0), c->Dim(c->input(9), 0), &unused_dhandle));
            //TF_RETURN_IF_ERROR(c->Merge(c->Dim(c->input(0), 1), c->Dim(c->input(9), 1), &unused_dhandle));
            //TF_RETURN_IF_ERROR(c->Merge(c->Dim(c->input(0), 2), c->Dim(c->input(3), 0), &unused_dhandle));
            //TF_RETURN_IF_ERROR(c->Merge(c->Dim(c->input(1), 0), c->Dim(c->input(2), 0), &unused_dhandle));
            //TF_RETURN_IF_ERROR(c->Merge(c->Dim(c->input(1), 0), c->Dim(c->input(10), 0), &unused_dhandle));
            //TF_RETURN_IF_ERROR(c->Merge(c->Dim(c->input(1), 1), c->Dim(c->input(2), 1), &unused_dhandle));
            //TF_RETURN_IF_ERROR(c->Merge(c->Dim(c->input(1), 1), c->Dim(c->input(10), 1), &unused_dhandle));
            //TF_RETURN_IF_ERROR(c->Merge(c->Dim(c->input(1), 2), c->Dim(c->input(5), 0), &unused_dhandle));
            //TF_RETURN_IF_ERROR(c->Merge(c->Dim(c->input(2), 0), c->Dim(c->input(10), 0), &unused_dhandle));
            //TF_RETURN_IF_ERROR(c->Merge(c->Dim(c->input(2), 1), c->Dim(c->input(10), 1), &unused_dhandle));
            //TF_RETURN_IF_ERROR(c->Merge(c->Dim(c->input(2), 2), c->Dim(c->input(7), 0), &unused_dhandle));
            //TF_RETURN_IF_ERROR(c->Merge(c->Dim(c->input(3), 1), c->Dim(c->input(4), 0), &unused_dhandle));
            //TF_RETURN_IF_ERROR(c->Merge(c->Dim(c->input(3), 1), c->Dim(c->input(5), 1), &unused_dhandle));
            //TF_RETURN_IF_ERROR(c->Merge(c->Dim(c->input(3), 1), c->Dim(c->input(6), 0), &unused_dhandle));
            //TF_RETURN_IF_ERROR(c->Merge(c->Dim(c->input(7), 1), c->Dim(c->input(8), 0), &unused_dhandle));
            //TF_RETURN_IF_ERROR(c->Merge(c->Dim(c->input(11), 0), c->Dim(c->input(12), 0), &unused_dhandle));
            //TF_RETURN_IF_ERROR(c->Merge(c->Dim(c->input(11), 0), c->Dim(c->input(13), 0), &unused_dhandle));
            //TF_RETURN_IF_ERROR(c->Merge(c->Dim(c->input(11), 0), c->Dim(c->input(14), 0), &unused_dhandle));

            //if (hidden_units > 1024) {
            //    return errors::InvalidArgument (
            //            "Hidden units should not be larger than 1024.");
            //}

            if (hidden_units % head_num != 0) {
                return errors::InvalidArgument (
                        "Hidden units should be multiple of head number.");
            }

            c->set_output(0, c->MakeShape({size_per_head}));
            c->set_output(1, c->MakeShape({size_per_head}));
            c->set_output(2, c->MakeShape({batch, seq_len, head_num*size_per_head}));

            return Status::OK();
    });

template <typename Device, typename T>
class LayernormGradOp : public CommonOp<T> {
 public:
  explicit LayernormGradOp(OpKernelConstruction* context): CommonOp<T>(context) {
      OP_REQUIRES_OK(context, context->GetAttr("head_num", &head_num_));
      OP_REQUIRES_OK(context, context->GetAttr("hidden_units", &hidden_units_));
      OP_REQUIRES_OK(context, context->GetAttr("batch", &batch_size_));
      OP_REQUIRES_OK(context, context->GetAttr("size_per_head", &size_per_head_));
      OP_REQUIRES_OK(context, context->GetAttr("alpha", &alpha_));
  }

  void Compute(OpKernelContext* context) override {

      int seq_len_;
      int out_units_;

      seq_len_ = context->input(0).dim_size(1);  // T
      out_units_ = context->input(0).dim_size(2);     // O

    // print the shape of all the inputs
    //std::cout << "from_tensor: ["
    //    << context->input(0).dim_size(0) <<", "
    //    << context->input(0).dim_size(1) <<", "
    //    << context->input(0).dim_size(2) << "]"
    //    << std::endl;

    //std::cout << "k_tensor: ["
    //    << context->input(1).dim_size(0) <<", "
    //    << context->input(1).dim_size(1) <<", "
    //    << context->input(1).dim_size(2) << "]"
    //    << std::endl;

    OP_REQUIRES(context, seq_len_ > 0 && batch_size_ > 0 && hidden_units_ &&
                size_per_head_ > 0 && head_num_ > 0,
                errors::InvalidArgument("Dimension for each input should be > 0"));

    int rank = (int)context->input(0).dims();
    OP_REQUIRES(context, rank==3,
                errors::InvalidArgument("Invalid rank. The rank of grads tensor should be 3\
                                        ([batch size, sequence length, hidden dimension]), but got ", rank));

    rank = (int)(context->input(1).dims());
    OP_REQUIRES(context, rank==3,
                errors::InvalidArgument("Invalid rank. The rank of K tensor should be 3\
                                        ([batch size, sequence length, hidden dimension]), but got ", rank));
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
//    rank = (int)(context->input(3).dims());
//    OP_REQUIRES(context, rank==2,
//                errors::InvalidArgument("Invalid rank. The rank of attr_q_kernel should be 2\
//                                        [hidden_dimension(C_q), hidden dimension(C)], but got ", rank));
//    OP_REQUIRES(context, (int)(context->input(3).dim_size(0)) == hidden_size_q_,
//                errors::InvalidArgument("Invalid shape: attr_q_kernel_shape[0] != C_q: ", context->input(3).dim_size(0), " vs ", hidden_size_q_));
//    OP_REQUIRES(context, (int)(context->input(3).dim_size(1)) == hidden_units_,
//                errors::InvalidArgument("Invalid shape: attr_q_kernel_shape[1] != C: ", context->input(3).dim_size(1), " vs ", hidden_units_));
//
//    rank = (int)(context->input(4).dims());
//    OP_REQUIRES(context, rank==1,
//                errors::InvalidArgument("Invalid rank. The rank of attr_q_bias should be 1\
//                                        ([hidden dimension(C)]), but got ", rank));
//    OP_REQUIRES(context, (int)(context->input(4).dim_size(0)) == hidden_units_,
//                errors::InvalidArgument("Invalid shape: attr_q_bias[0] != C: ", context->input(4).dim_size(0), " vs ", hidden_units_));
//
//    rank = (int)(context->input(5).dims());
//    OP_REQUIRES(context, rank==2,
//                errors::InvalidArgument("Invalid rank. The rank of attr_k_kernel should be 2\
//                                        ([hidden_dimension(C), hidden dimension(C)]), but got ", rank));
//    OP_REQUIRES(context, ((int)(context->input(5).dim_size(0)) == hidden_size_k_) &&
//                         ((int)(context->input(5).dim_size(1)) == hidden_units_),
//                errors::InvalidArgument("Invalid shape: attr_k_kerne_shapel != [C_k, C]", context->input(5).shape().DebugString()));
//
//    rank = (int)(context->input(6).dims());
//    OP_REQUIRES(context, rank==1,
//                errors::InvalidArgument("Invalid rank. The rank of attr_k_bias should be 1\
//                                        ([hidden dimension(C)]), but got ", rank));
//    OP_REQUIRES(context, (int)(context->input(6).dim_size(0)) == hidden_units_,
//                errors::InvalidArgument("Invalid shape: k_bias[0] != C: ", context->input(6).shape().DebugString()));
//
//    rank = (int)(context->input(7).dims());
//    OP_REQUIRES(context, rank==2,
//                errors::InvalidArgument("Invalid rank. The rank of attr_k_kernel should be 2\
//                                        ([hidden_dimension(C), hidden dimension(C)]), but got ", rank));
//    OP_REQUIRES(context, ((int)(context->input(7).dim_size(0)) == hidden_size_v_) &&
//                         ((int)(context->input(7).dim_size(1)) == out_units_),
//                errors::InvalidArgument("Invalid shape: attr_v_kerne_shapel != [C_v, O]: ", context->input(7).shape().DebugString()));
//
//    rank = (int)(context->input(8).dims());
//    OP_REQUIRES(context, rank==1,
//                errors::InvalidArgument("Invalid rank. The rank of attr_k_bias should be 1\
//                                        ([hidden dimension(C)]), but got ", rank));
//    OP_REQUIRES(context, (int)(context->input(8).dim_size(0)) == out_units_ && out_units_ % size_per_head_ == 0,
//                errors::InvalidArgument("Invalid shape: v_bias[0] != O, or out_units is not multiple of ize_per_head", context->input(8).shape().DebugString()));

//    rank = (int)(context->input(9).dims());
//    OP_REQUIRES(context, rank==2,
//            errors::InvalidArgument("Invalid rank. The rank of q_mask should be 2\
//                ([batch(N), seq_len(T_q)]), but got ", rank));
//    OP_REQUIRES(context, ((int)(context->input(9).dim_size(0)) == batch_size_) &&
//            ((int)(context->input(9).dim_size(1)) == from_seq_len_),
//            errors::InvalidArgument("Invalid shape: q_mask_shape != [N, T_q]: ", context->input(9).shape().DebugString()));
//
//    rank = (int)(context->input(10).dims());
//    OP_REQUIRES(context, rank==2,
//                errors::InvalidArgument("Invalid rank. The rank of k_mask should be 2\
//                                        ([batch(M), seq_len(T_k)]), but got ", rank));
//    OP_REQUIRES(context, (context->input(10).dim_size(0) == batch_size_kv_) &&
//            (context->input(10).dim_size(1) == to_seq_len_),
//            errors::InvalidArgument("Invalid shape: k_mask_shape != [M, T_k]: ", context->input(10).shape().DebugString()));
//
//    if (do_layer_norm_)
//    {
//        rank = (int)(context->input(11).dims());
//        OP_REQUIRES(context, rank==1,
//                errors::InvalidArgument("Invalid rank. The rank of ln_q_beta should be 1\
//                    ([size_per_head]), but got ", rank));
//        OP_REQUIRES(context, (int)(context->input(11).dim_size(0)) == size_per_head_,
//                errors::InvalidArgument("Invalid shape: ln_q_beta_shape != [size_per_head_]: ", context->input(11).shape().DebugString()));
//
//        rank = (int)(context->input(12).dims());
//        OP_REQUIRES(context, rank==1,
//                errors::InvalidArgument("Invalid rank. The rank of ln_q_gamma should be 1\
//                    ([size_per_head]), but got ", rank));
//        OP_REQUIRES(context, (int)(context->input(12).dim_size(0)) == size_per_head_,
//                errors::InvalidArgument("Invalid shape: ln_q_gamma_shape != [size_per_head_]: ", context->input(12).shape().DebugString()));
//
//        rank = (int)(context->input(13).dims());
//        OP_REQUIRES(context, rank==1,
//                errors::InvalidArgument("Invalid rank. The rank of ln_k_beta should be 1\
//                    ([size_per_head]), but got ", rank));
//        OP_REQUIRES(context, (int)(context->input(13).dim_size(0)) == size_per_head_,
//                errors::InvalidArgument("Invalid shape: ln_k_beta_shape != [size_per_head_]: ", context->input(13).shape().DebugString()));
//
//        rank = (int)(context->input(14).dims());
//        OP_REQUIRES(context, rank==1,
//                errors::InvalidArgument("Invalid rank. The rank of ln_k_gamma should be 1\
//                    ([size_per_head]), but got ", rank));
//        OP_REQUIRES(context, (int)(context->input(14).dim_size(0)) == size_per_head_,
//                errors::InvalidArgument("Invalid shape: ln_k_beta_shape != [size_per_head_]: ", context->input(14).shape().DebugString()));
//    }

    OP_REQUIRES(context, context->num_inputs() == 5, errors::InvalidArgument("Less input arguments"));

    Tensor * d_gamma_ = nullptr;
    OP_REQUIRES_OK(
            context,
            context->allocate_output(0, {size_per_head_}, &d_gamma_));

    Tensor * d_betta_ = nullptr;
    OP_REQUIRES_OK(
            context,
            context->allocate_output(1, {size_per_head_}, &d_betta_));

    Tensor * d_x_ = nullptr;
    OP_REQUIRES_OK(
            context,
            context->allocate_output(2, {batch_size_, seq_len_, head_num_ * size_per_head_}, &d_x_));

//    Tensor * d_tmp_ = nullptr;
//    OP_REQUIRES_OK(
//            context,
//            context->allocate_output(3, {batch_size_, seq_len_, head_num_ * size_per_head_}, &d_tmp_));

    if (d_gamma_->NumElements() == 0)
        return;
    if (d_betta_->NumElements() == 0)
        return;
    if (d_x_->NumElements() == 0)
        return;


    layernormbackprop::LayernormGradParam<DataType_> param;
    param.op_context = context;
    param.alpha = alpha_;
    //param.cublas_handle = this->get_cublas_handler();
    //check_cuda_error(cublasSetStream(param.cublas_handle, param.stream));
    this->get_tensor(context, 0, &param.grads);
    this->get_tensor(context, 1, &param.x_data);
    this->get_tensor(context, 2, &param.vars);
    this->get_tensor(context, 3, &param.means);
    this->get_tensor(context, 4, &param.gamma);

    param.d_gamma = reinterpret_cast<DataType_ *>(d_gamma_->flat<T>().data());
    param.d_betta = reinterpret_cast<DataType_ *>(d_betta_->flat<T>().data());
    param.d_x = reinterpret_cast<DataType_ *>(d_x_->flat<T>().data());
    //param.d_tmp = reinterpret_cast<DataType_ *>(d_tmp_->flat<T>().data());

    functor::LayernormGradOpFunctor<Device, T, DataType_>::Compute(
            context,
            param,
            batch_size_,
            seq_len_,
            head_num_,
            hidden_units_,
            size_per_head_
            );
  }
 private:
  int head_num_, hidden_units_;
  int batch_size_, size_per_head_;
  float alpha_;
  typedef TFTraits<T> traits_;
  typedef typename traits_::DataType DataType_;
};

#ifdef GOOGLE_CUDA

#define REGISTER_GPU(T)                 \
    REGISTER_KERNEL_BUILDER(            \
        Name("LayernormGrad")      \
        .Device(DEVICE_GPU)             \
        .TypeConstraint<T>("T"),        \
      LayernormGradOp<GPUDevice, T>);

REGISTER_GPU(float);
//REGISTER_GPU(double);
//REGISTER_GPU(Eigen::half);

#undef REGISTER_GPU

#endif

} // end namespace

} // end namespace tensorflow
