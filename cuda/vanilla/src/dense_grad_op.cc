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

REGISTER_OP("DenseGrad")
    .Input("q_grads: T")           // 0 [N,T_q,C]
    .Input("k_grads: T")           // 1 [N,T_k,C]
    .Input("v_grads: T")           // 2 [N,T_k,O]
    .Input("query: T")             // 3 [N,T_q,C_q]
    .Input("key: T")               // 4 [N,T_k,C_k]
    .Input("value: T")             // 5 [N,T_k,C_v]
    .Input("q_kernel: T")          // 6 [C_q,C]
    .Input("k_kernel: T")          // 7 [C_k,C]
    .Input("v_kernel: T")          // 8 [C_v,O]
    .Input("query_layer: T")       // 9 [h*N, T_q, C/h]
    .Input("key_layer: T")         // 10 [h*N, T_k, C/h]
    .Input("value_layer: T")       // 11 [h*N, T_k, O/h]
    .Output("dq: T")               // 0 [N, T_q, C_q]
    .Output("dk: T")               // 1 [N, T_k, C_k]
    .Output("dv: T")               // 2 [N, T_k, C_v]
    .Output("dwq: T")              // 3 [C_q, C]
    .Output("dbq: T")              // 4 [C]
    .Output("dwk: T")              // 5 [C_k, C]
    .Output("dbk: T")              // 6 [C]
    //.Output("dwv: T")              // 7 [C_v, O]
    .Output("dwv: T")              // 7 [N, C_v, O], intermediate and do reduce_sum outside
    .Output("dbv: T")              // 8 [O]
    //.Output("dtmp: T")             // 9 [N, T_k, C]
    .Attr("T: {float, half}")
    .Attr("head_num: int >= 1")
    .Attr("hidden_units: int >= 1")
    .Attr("batch: int >= 1")
    .Attr("size_per_head: int >= 1")
    .Attr("size_per_head_out: int >= 1")
    .Attr("lrelu_alpha: float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
            int head_num;
            c->GetAttr("head_num", &head_num);
            int hidden_units;
            c->GetAttr("hidden_units", &hidden_units);
            int batch;
            c->GetAttr("batch", &batch);
            int size_per_head;
            c->GetAttr("size_per_head", &size_per_head);
            int size_per_head_out;
            c->GetAttr("size_per_head_out", &size_per_head_out);
            int hidden_units_out = head_num * size_per_head_out;

            //auto batch = c->Dim(c->input(0), 0);
            auto from_seq_len = c->Dim(c->input(0), 1);
            auto to_seq_len = c->Dim(c->input(1), 1);
            auto hs_q = c->Dim(c->input(3), 2);
            auto hs_k = c->Dim(c->input(4), 2);
            auto hs_v = c->Dim(c->input(5), 2);

            // add parameters checking
            shape_inference::ShapeHandle unused_handle;
            TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &unused_handle));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &unused_handle));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 3, &unused_handle));
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

            shape_inference::DimensionHandle unused_dhandle;
            TF_RETURN_IF_ERROR(c->Merge(c->Dim(c->input(0), 1), c->Dim(c->input(3), 1), &unused_dhandle));
            TF_RETURN_IF_ERROR(c->Merge(c->Dim(c->input(1), 1), c->Dim(c->input(4), 1), &unused_dhandle));
            TF_RETURN_IF_ERROR(c->Merge(c->Dim(c->input(2), 1), c->Dim(c->input(5), 1), &unused_dhandle));
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

            c->set_output(0, c->MakeShape({batch, from_seq_len, hs_q}));
            c->set_output(1, c->MakeShape({batch, to_seq_len, hs_k}));
            c->set_output(2, c->MakeShape({batch, to_seq_len, hs_v}));
            c->set_output(3, c->MakeShape({hs_q, hidden_units}));
            c->set_output(4, c->MakeShape({hidden_units}));
            c->set_output(5, c->MakeShape({hs_k, hidden_units}));
            c->set_output(6, c->MakeShape({hidden_units}));
            //c->set_output(7, c->MakeShape({hs_v, hidden_units_out}));
            c->set_output(7, c->MakeShape({batch, hs_v, hidden_units_out}));
            c->set_output(8, c->MakeShape({hidden_units_out}));
            //c->set_output(9, c->MakeShape({batch, to_seq_len, hidden_units_out}));

            return Status::OK();
    });

template <typename Device, typename T>
class DenseGradOp : public CommonOp<T> {
 public:
  explicit DenseGradOp(OpKernelConstruction* context): CommonOp<T>(context) {
      OP_REQUIRES_OK(context, context->GetAttr("head_num", &head_num_));
      OP_REQUIRES_OK(context, context->GetAttr("hidden_units", &hidden_units_));
      OP_REQUIRES_OK(context, context->GetAttr("batch", &batch_size_));
      OP_REQUIRES_OK(context, context->GetAttr("size_per_head", &size_per_head_));
      OP_REQUIRES_OK(context, context->GetAttr("size_per_head_out", &size_per_head_out_));
      OP_REQUIRES_OK(context, context->GetAttr("lrelu_alpha", &lrelu_alpha_));
  }

  void Compute(OpKernelContext* context) override {

      int from_seq_len_ = context->input(0).dim_size(1);  // T
      int to_seq_len_ = context->input(1).dim_size(1);    // T
      int hs_q_ = context->input(3).dim_size(2);          // C_q
      int hs_k_ = context->input(4).dim_size(2);          // C_k
      int hs_v_ = context->input(5).dim_size(2);          // C_v
      int hidden_units_out_ = size_per_head_out_ * head_num_; // O

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

    OP_REQUIRES(context, from_seq_len_ > 0 && batch_size_ > 0 && hidden_units_ &&
                size_per_head_ > 0 && head_num_ > 0 && to_seq_len_,
                errors::InvalidArgument("Dimension for each input should be > 0"));

    int rank = (int)context->input(0).dims();
    OP_REQUIRES(context, rank==3,
                errors::InvalidArgument("Invalid rank. The rank of q_grads tensor should be 3\
                                        ([batch size, sequence length, hidden dimension]), but got ", rank));

    rank = (int)context->input(1).dims();
    OP_REQUIRES(context, rank==3,
                errors::InvalidArgument("Invalid rank. The rank of k_grads tensor should be 3\
                                        ([batch size, sequence length, hidden dimension]), but got ", rank));

    rank = (int)context->input(2).dims();
    OP_REQUIRES(context, rank==3,
                errors::InvalidArgument("Invalid rank. The rank of v_grads tensor should be 3\
                                        ([batch size, sequence length, hidden dimension]), but got ", rank));

    OP_REQUIRES(context, context->num_inputs() == 12, errors::InvalidArgument("Less input arguments"));

    //printf("########################################################a\n");
    //printf("PRINTED from %s, line %d\n", __FILE__, __LINE__);
    //printf("########################################################a\n");
    //printf("batch = %d\n", batch_size_);
    //printf("from_seq_len_ = %d\n", from_seq_len_);
    //printf("to_seq_len_ = %d\n", to_seq_len_);
    //printf("hidden_size_q = %d\n", hs_q_);
    //printf("hidden_size_k = %d\n", hs_k_);
    //printf("hidden_size_v = %d\n", hs_v_);
    //printf("########################################################a\n");

    Tensor * dq_ = nullptr;
    OP_REQUIRES_OK(
            context,
            context->allocate_output(0, {batch_size_, from_seq_len_, hs_q_}, &dq_));

    Tensor * dk_ = nullptr;
    OP_REQUIRES_OK(
            context,
            context->allocate_output(1, {batch_size_, to_seq_len_, hs_k_}, &dk_));

    Tensor * dv_ = nullptr;
    OP_REQUIRES_OK(
            context,
            context->allocate_output(2, {batch_size_, to_seq_len_, hs_v_}, &dv_));

    Tensor * dwq_ = nullptr;
    OP_REQUIRES_OK(
            context,
            context->allocate_output(3, {hs_q_, hidden_units_}, &dwq_));

    Tensor * dbq_ = nullptr;
    OP_REQUIRES_OK(
            context,
            context->allocate_output(4, {hidden_units_}, &dbq_));

    Tensor * dwk_ = nullptr;
    OP_REQUIRES_OK(
            context,
            context->allocate_output(5, {hs_k_, hidden_units_}, &dwk_));

    Tensor * dbk_ = nullptr;
    OP_REQUIRES_OK(
            context,
            context->allocate_output(6, {hidden_units_}, &dbk_));

//    Tensor * dwv_ = nullptr;
//    OP_REQUIRES_OK(
//            context,
//            context->allocate_output(7, {hs_v_, hidden_units_out_}, &dwv_));

    Tensor * dwv_ = nullptr;
    OP_REQUIRES_OK(
            context,
            context->allocate_output(7, {batch_size_, hs_v_, hidden_units_out_}, &dwv_));

    Tensor * dbv_ = nullptr;
    OP_REQUIRES_OK(
            context,
            context->allocate_output(8, {hidden_units_out_}, &dbv_));

    if (dq_->NumElements() == 0)
        return;
    if (dk_->NumElements() == 0)
        return;
    if (dv_->NumElements() == 0)
        return;

    densebackprop::DenseGradParam<DataType_> param;
    param.op_context = context;
    param.alpha = lrelu_alpha_;
    //param.cublas_handle = this->get_cublas_handler();
    //check_cuda_error(cublasSetStream(param.cublas_handle, param.stream));
    this->get_tensor(context, 0, &param.q_grads);
    this->get_tensor(context, 1, &param.k_grads);
    this->get_tensor(context, 2, &param.v_grads);
    this->get_tensor(context, 3, &param.query);
    this->get_tensor(context, 4, &param.key);
    this->get_tensor(context, 5, &param.value);
    this->get_tensor(context, 6, &param.q_kernel);
    this->get_tensor(context, 7, &param.k_kernel);
    this->get_tensor(context, 8, &param.v_kernel);
    this->get_tensor(context, 9, &param.query_layer);
    this->get_tensor(context, 10, &param.key_layer);
    this->get_tensor(context, 11, &param.value_layer);

    param.dq = reinterpret_cast<DataType_ *>(dq_->flat<T>().data());
    param.dk = reinterpret_cast<DataType_ *>(dk_->flat<T>().data());
    param.dv = reinterpret_cast<DataType_ *>(dv_->flat<T>().data());
    param.dwq = reinterpret_cast<DataType_ *>(dwq_->flat<T>().data());
    param.dbq = reinterpret_cast<DataType_ *>(dbq_->flat<T>().data());
    param.dwk = reinterpret_cast<DataType_ *>(dwk_->flat<T>().data());
    param.dbk = reinterpret_cast<DataType_ *>(dbk_->flat<T>().data());
    //param.dwv = reinterpret_cast<DataType_ *>(dwv_->flat<T>().data());
    param.dbv = reinterpret_cast<DataType_ *>(dbv_->flat<T>().data());
    param.dv_inter = reinterpret_cast<DataType_ *>(dwv_->flat<T>().data());

    functor::DenseGradOpFunctor<Device, T, DataType_>::Compute(
            context,
            param,
            batch_size_,
            from_seq_len_,
            to_seq_len_,
            head_num_,
            hidden_units_,
            size_per_head_,
            size_per_head_out_,
            hs_q_,
            hs_k_,
            hs_v_
            );
  }
 private:
  int head_num_, hidden_units_;
  int batch_size_, size_per_head_, size_per_head_out_;
  float lrelu_alpha_;
  typedef TFTraits<T> traits_;
  typedef typename traits_::DataType DataType_;
};

#ifdef GOOGLE_CUDA

#define REGISTER_GPU(T)                 \
    REGISTER_KERNEL_BUILDER(            \
        Name("DenseGrad")      \
        .Device(DEVICE_GPU)             \
        .TypeConstraint<T>("T"),        \
      DenseGradOp<GPUDevice, T>);

REGISTER_GPU(float);
//REGISTER_GPU(double);
//REGISTER_GPU(Eigen::half);

#undef REGISTER_GPU

#endif

} // end namespace

} // end namespace tensorflow
