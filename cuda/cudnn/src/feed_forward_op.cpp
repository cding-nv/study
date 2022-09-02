#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "common_op.h"
#include "feed_forward.h"

namespace tensorflow {

namespace {

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("FeedForward")
    .Input("from_tensor: T")       // 0 [N,T_q,C]
    .Input("attr_q_kernel_0: T")   // 2 [C,C]
    .Input("attr_q_bias_0: T")     // 3 [C]
    .Output("output: T")           // 0: [h*N,dk0,T_q]
    .Attr("T: {float, half}")
    .Attr("head_num: int >= 1")
    .Attr("D_Q_0: int >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
            int head_num;
            c->GetAttr("head_num", &head_num);
            //int DK0;
            //c->GetAttr("D_K_0", &DK0);
            int DQ0;
            c->GetAttr("D_Q_0", &DQ0);
            int DQ1;
            //c->GetAttr("D_Q_1", &DQ1);

            auto batch = c->Dim(c->input(0), 0);
            auto from_seq_len = c->Dim(c->input(0), 1);
            //auto to_seq_len = c->Dim(c->input(1), 1);

            c->set_output(0, c->MakeShape({batch, head_num, DQ0, from_seq_len}));  // matmul

            return Status::OK();
    });

template <typename Device, typename T>
class FeedForwardOp : public CommonOp<T> {
 public:
  explicit FeedForwardOp(OpKernelConstruction* context): CommonOp<T>(context) {
      std::cout << "### start of REGISTER_OP FeedForwardOp init" << std::endl;
      OP_REQUIRES_OK(context, context->GetAttr("head_num", &head_num_));
      OP_REQUIRES_OK(context, context->GetAttr("D_Q_0", &D_Q_0_));
      std::cout << "### end of REGISTER_OP FeedForwardOp init" << std::endl;
  }

  void Compute(OpKernelContext* context) override {
    std::cout << "### start of FeedForwardOp compute" << std::endl;
      int rank = (int)context->input(0).dims();
    OP_REQUIRES(context, rank==3,
                errors::InvalidArgument("Invalid rank. The rank of from tensor should be 3\
                                        ([batch size, sequence length, hidden dimension])"));

    int batch_n_ = (int)context->input(0).dim_size(0);    // N
    int from_seq_len_ = (int)context->input(0).dim_size(1);  // T_q
    int hidden_units = (int)context->input(0).dim_size(2);// C

    std::vector<std::array<int, 3>> gemm_algos;
    gemm_algos.push_back(std::array<int, 3>({99, 99, 99}));
    gemm_algos.push_back(std::array<int, 3>({99, 99, 99}));
    gemm_algos.push_back(std::array<int, 3>({99, 99, 99}));
    gemm_algos.push_back(std::array<int, 3>({99, 99, 99}));
    gemm_algos.push_back(std::array<int, 3>({99, 99, 99}));

    std::cout << "batch_n_,from_seq_len_,hidden_units = " << batch_n_  << "," << from_seq_len_ << "," << hidden_units << std::endl;
    FeedForward<T> _qkv_linear = (typename FeedForward<T>::Config(1,
                                                  8,
                                                  8,
                                                  gemm_algos[0]));

    OP_REQUIRES(context, context->num_inputs() == 3, errors::InvalidArgument("Less input arguments"));

    //const int hidden_units_ = head_num_ * size_per_head_;
    //OP_REQUIRES(context, hidden_units_ == hidden_units, errors::InvalidArgument("hidden size of input != head_num * size_per_head."));

    cuda::MultiHeadInitParam<DataType_> param;
    param.cublas_handle = this->get_cublas_handler();
    check_cuda_error(cublasSetStream(param.cublas_handle, param.stream));
    param.op_context = context;
    this->get_tensor(context, 0, &param.from_tensor);
    //this->get_tensor(context, 1, &param.to_tensor);
    this->get_tensor(context, 2, &param.self_attention.query_weight_0.kernel);

    int valid_word_num = batch_n_* from_seq_len_;
    param.valid_word_num = valid_word_num;
    param.sequence_id_offset = nullptr;

    Tensor * output = nullptr;
    std::cout << "head_num_,batch_n_,from_seq_len_,D_Q_0_ = "
              << head_num_  << "," << batch_n_ << "," << from_seq_len_ << "," << D_Q_0_ << std::endl;
    OP_REQUIRES_OK(
            context,
            context->allocate_output(0, {batch_n_, from_seq_len_, head_num_ * D_Q_0_}, &output)); // matmul
            //context->allocate_output(0, {batch_n_, head_num_, to_seq_len_, D_K_0_}, &output)); // multiply
    param.attr_out = reinterpret_cast<DataType_ *>(output->flat<T>().data());

    int bsz_seq = batch_n_* from_seq_len_;

    #if 1
    std::cout << "### before qkv_linear forward" << std::endl;
    std::cout << "### bsz_seq=" << bsz_seq << std::endl;
    _qkv_linear.Forward(bsz_seq,
                        param.from_tensor,
                        param.self_attention.query_weight_0.kernel,
                        param.attr_out,
                        param.cublas_handle);
    std::cout << "### end qkv_linear forward" << std::endl;

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
  int head_num_;
  int D_Q_0_, D_Q_1_, D_K_0_;
  typedef TFTraits<T> traits_;
  typedef typename traits_::DataType DataType_;
  //FeedForward<T> _qkv_linear;
};

#ifdef GOOGLE_CUDA

#define REGISTER_GPU(T)                 \
    REGISTER_KERNEL_BUILDER(            \
        Name("FeedForward")      \
        .Device(DEVICE_GPU)             \
        .TypeConstraint<T>("T"),        \
      FeedForwardOp<GPUDevice, T>);

REGISTER_GPU(float);
//REGISTER_GPU(Eigen::half);

#undef REGISTER_GPU

#endif

} // end namespace

} // end namespace tensorflow
