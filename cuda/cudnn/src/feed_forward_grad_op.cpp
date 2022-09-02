#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "common_op.h"

#include "feed_forward.h"

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

namespace {

REGISTER_OP("FeedForwardGrad")
    .Input("grad_key: T")           // 0 [h*N,S,dk0]
    .Input("key: T")                // 1 [M,S,L]
    .Input("w0: T")                 // 2 [L, h*DQ1]
    .Input("b0: T")                 // 3 [h*DQ1]
    .Input("indices: int32")          // 4 [N]
    .Output("d_key: T")             // 0 [M,S,L]
    .Output("d_w: T")               // 1 [L,h*DQ1]
    .Output("d_b: T")               // 2 [h*DQ1]
    //.Output("inter: T")             // 3 []
    .Attr("T: {float, half}")
    .Attr("head_num: int >= 1")
    //.Attr("hidden_size: int >= 1")
    .Attr("D_Q_0: int >= 1")
    .Attr("D_Q_1: int >= 1")
    .Attr("D_K_0: int >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {

            int DK0;
            c->GetAttr("D_K_0", &DK0);
            int DQ0;
            c->GetAttr("D_Q_0", &DQ0);
            int DQ1;
            c->GetAttr("D_Q_1", &DQ1);
            int head_num;
            c->GetAttr("head_num", &head_num);
            //int hidden_size;
            //c->GetAttr("hidden_size", &hidden_size);

            auto batch_m = c->Dim(c->input(1), 0);
            auto batch_n = c->Dim(c->input(4), 0);
            auto from_seq_len = c->Dim(c->input(1), 1);
            auto to_seq_len = c->Dim(c->input(2), 1);

            c->set_output(0, c->input(1));
            c->set_output(1, c->input(2));
            c->set_output(2, c->input(3));
            //c->set_output(3, c->MakeShape({hidden_size, head_num, DK0}));

            return Status::OK();
    });

template <typename Device, typename T>
class FeedForwardGradOp : public CommonOp<T> {
 public:
  explicit FeedForwardGradOp(OpKernelConstruction* context): CommonOp<T>(context) {
      OP_REQUIRES_OK(context, context->GetAttr("head_num", &head_num_));
      //OP_REQUIRES_OK(context, context->GetAttr("hidden_size", &hidden_size_));
      OP_REQUIRES_OK(context, context->GetAttr("D_Q_0", &D_Q_0_));
      OP_REQUIRES_OK(context, context->GetAttr("D_Q_1", &D_Q_1_));
      OP_REQUIRES_OK(context, context->GetAttr("D_K_0", &D_K_0_));
  }

  void Compute(OpKernelContext* context) override {

    int hN = (int)context->input(0).dim_size(0);        // h * N
    int hidden_size_ = (int)context->input(1).dim_size(2); // L
    //from_seq_len_ = (int)context->input(1).dim_size(1);  // 1
    int from_seq_len_ = 1;
    to_seq_len_ = (int)context->input(0).dim_size(1);    // S
    int batch_m_ = (int)context->input(1).dim_size(0);   // M
    int batch_n_ = (int)context->input(4).dim_size(0);   // N
    //batch_m_ = hN / head_num_;

    std::cout << "grad_key: ["
        << context->input(0).dim_size(0) <<", "
        << context->input(0).dim_size(1) <<", "
        << context->input(0).dim_size(2) << "]"
        << std::endl;

    std::cout << "key: ["
        << context->input(1).dim_size(0) <<", "
        << context->input(1).dim_size(1) <<", "
        << context->input(1).dim_size(2) << "]"
        << std::endl;

    std::cout << "w0: ["
        << context->input(2).dim_size(0) <<", "
        << context->input(2).dim_size(1) << "]"
        << std::endl;

    std::cout << "b0: ["
        << context->input(3).dim_size(0) << "]"
        << std::endl;

    std::cout << "indices: ["
        << context->input(4).dim_size(0) << "]"
        << std::endl;

    std::cout << "head_num: " << head_num_ <<std::endl;
    //std::cout << "hidden_size: " << hidden_size_ <<std::endl;
    std::cout << "D_Q_0: " << D_Q_0_ <<std::endl;
    std::cout << "D_Q_1: " << D_Q_1_ <<std::endl;
    std::cout << "D_K_0: " << D_K_0_ <<std::endl;

    std::vector<std::array<int, 3>> gemm_algos;
    gemm_algos.push_back(std::array<int, 3>({99, 99, 99}));
    gemm_algos.push_back(std::array<int, 3>({99, 99, 99}));
    gemm_algos.push_back(std::array<int, 3>({99, 99, 99}));
    gemm_algos.push_back(std::array<int, 3>({99, 99, 99}));
    gemm_algos.push_back(std::array<int, 3>({99, 99, 99}));

    //std::cout << "batch_n_,from_seq_len_,hidden_units = " << batch_n_  << "," << from_seq_len_ << "," << hidden_units << std::endl;
    FeedForward<T> _qkv_linear = (typename FeedForward<T>::Config(hN,
                                                  to_seq_len_, //outputSize 8
                                                  hidden_size_, // inputSize 3
                                                  gemm_algos[0]));
    std::cout << "FeedForward config batchsize/outputsize/inputsize: "
            << hN << "/" << to_seq_len_ << "/" << hidden_size_ << std::endl;

    //OP_REQUIRES(context, context->num_inputs() == 8, errors::InvalidArgument("Less input arguments"));
    //keybackprop::KeyGradParam<DataType_> param;
    cuda::FeedForwardGradParam<DataType_> param;
    param.cublas_handle = this->get_cublas_handler();
    check_cuda_error(cublasSetStream(param.cublas_handle, param.stream));
    param.op_context = context;
  
    this->get_tensor(context, 0, &param.grad_key);
    this->get_tensor(context, 1, &param.key);
    this->get_tensor(context, 2, &param.w0);
    this->get_tensor(context, 3, &param.b0);

    Tensor * out_dk = nullptr;
    OP_REQUIRES_OK(
            context,
            context->allocate_output(0, {batch_m_, to_seq_len_, hidden_size_}, &out_dk));
    std::cout << "### batch_m_, to_seq_len_, hidden_size_ = " << batch_m_ << "/" << to_seq_len_ << "/" << hidden_size_ << std::endl;
    param.d_key = reinterpret_cast<DataType_ *>(out_dk->flat<T>().data());

    Tensor * out_dw0 = nullptr;
    OP_REQUIRES_OK(
            context,
            context->allocate_output(1, {hidden_size_, head_num_ * D_Q_1_}, &out_dw0));
    std::cout << "### hidden_size_, head_num_ , D_Q_1_" << hidden_size_ << "/" << head_num_ << "/" << D_Q_1_ << std::endl;
    param.d_w0 = reinterpret_cast<DataType_ *>(out_dw0->flat<T>().data());

    Tensor * out_db0 = nullptr;
    OP_REQUIRES_OK(
            context,
            context->allocate_output(2, {head_num_ * D_Q_1_}, &out_db0));
    param.d_b0 = reinterpret_cast<DataType_ *>(out_db0->flat<T>().data());

    _qkv_linear.Backward(batch_m_,
                         param.grad_key,   // out_grad
                         param.key, // x, feature
                         param.w0, // weights
                         param.d_w0, // dw output
                         param.d_b0, // db output
                         param.cublas_handle,
                         param.stream,
                         param.d_key);   // dx output
    std::cout << "### end qkv_linear backward" << std::endl;

    #if 0
    float* h_mem = reinterpret_cast<float *>(malloc(8 * sizeof(float)));
    check_cuda_error(cudaMemcpy(h_mem, param.attr_out, 8 * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 8; i++) {
        std::cout << "attr_out### " << h_mem[i] << std::endl;
    }
    check_cuda_error(cudaMemcpy(h_mem, param.from_tensor, 8* sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 8; i++) {
        std::cout << "from_tensor### " << h_mem[i] << std::endl;
    }

    float* h_mem_64 = reinterpret_cast<float *>(malloc(64 * sizeof(float)));
    check_cuda_error(cudaMemcpy(h_mem_64, param.self_attention.query_weight_0.kernel, 64 * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 64; i++) {
        std::cout << "weight### " << h_mem_64[i] << std::endl;
    }
    #endif
  }
 private:
  int from_seq_len_, to_seq_len_, head_num_;
  int D_Q_0_, D_Q_1_, D_K_0_;
  typedef TFTraits<T> traits_;
  typedef typename traits_::DataType DataType_;
};

#ifdef GOOGLE_CUDA

#define REGISTER_GPU(T)                 \
    REGISTER_KERNEL_BUILDER(            \
        Name("FeedForwardGrad")      \
        .Device(DEVICE_GPU)             \
        .TypeConstraint<T>("T"),        \
      FeedForwardGradOp<GPUDevice, T>);

REGISTER_GPU(float);
//REGISTER_GPU(double);
//REGISTER_GPU(Eigen::half);

#undef REGISTER_GPU

#endif

} // end namespace FeedForwardGradGrad

} // end namespace tensorflow
