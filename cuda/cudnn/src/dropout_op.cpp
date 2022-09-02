#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "common_op.h"
#include "dropout.h"

namespace tensorflow {

namespace {

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("DropOut")
    .Input("from_tensor: T")
    .Output("output: T")
    .Attr("T: {float, half}")
    .Attr("prob: float")
    .Attr("batch: int >= 1")
    .Attr("heads: int >= 1")
    .Attr("seq_len: int >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
            int batch_;
            int head_num;
            int seq_len_;
            float prob_;
            c->GetAttr("prob", &prob_);
            c->GetAttr("batch", &batch_);
            c->GetAttr("heads", &head_num);
            c->GetAttr("seq_len", &seq_len_);
            c->set_output(0, c->MakeShape({batch_, head_num, seq_len_}));

            return Status::OK();
    });

template <typename Device, typename T>
class DropOutOp : public CommonOp<T> {
 public:
  explicit DropOutOp(OpKernelConstruction* context): CommonOp<T>(context) {
      OP_REQUIRES_OK(context, context->GetAttr("prob", &prob_));
      OP_REQUIRES_OK(context, context->GetAttr("heads", &heads_));
      OP_REQUIRES_OK(context, context->GetAttr("batch", &batch_));
      OP_REQUIRES_OK(context, context->GetAttr("seq_len", &seq_len_));
      check_cuda_error(cudaMalloc(&attn_prob_dropout_mask_ptr, batch_ * heads_ * seq_len_ * seq_len_ * sizeof(uint8_t)));
  }

  void Compute(OpKernelContext* context) override {
    std::cout << "### Peak start of DropOutOp compute" << std::endl;
    int rank = (int)context->input(0).dims();
    OP_REQUIRES(context, rank==3,
                errors::InvalidArgument("Invalid rank. The rank of from tensor should be 3\
                                        ([batch, heads, seq_len])"));

    std::cout << "batch_, heads_, seq_len_, prob_ = "
        << batch_ << "," << heads_ << "," << seq_len_ << "," << prob_ << std::endl;
    Dropout<T> _attn_prob_dropout = (typename Dropout<T>::Config(prob_, seq_len_));
    _attn_prob_dropout.SetTrainingMode(true);

    OP_REQUIRES(context, context->num_inputs() == 1, errors::InvalidArgument("Less input arguments"));

    cuda::DropoutParam<DataType_> param;
    param.cublas_handle = this->get_cublas_handler();
    check_cuda_error(cublasSetStream(param.cublas_handle, param.stream));
    param.op_context = context;
    this->get_tensor(context, 0, &param.from_tensor);

    Tensor * output = nullptr;
    OP_REQUIRES_OK(
            context,
            context->allocate_output(0, {batch_, heads_, seq_len_}, &output));
    param.attr_out = reinterpret_cast<DataType_ *>(output->flat<T>().data());

    _attn_prob_dropout.SetMask((uint8_t*)attn_prob_dropout_mask_ptr);
    _attn_prob_dropout.Forward(batch_ * heads_ * seq_len_, param.attr_out, param.from_tensor, param.stream);

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
  int heads_;
  int batch_;
  int seq_len_;
  float prob_;
  void* attn_prob_dropout_mask_ptr;
  typedef TFTraits<T> traits_;
  typedef typename traits_::DataType DataType_;
  //Dropout<T> _dropout;
};

#ifdef GOOGLE_CUDA

#define REGISTER_GPU(T)                 \
    REGISTER_KERNEL_BUILDER(            \
        Name("DropOut")      \
        .Device(DEVICE_GPU)             \
        .TypeConstraint<T>("T"),        \
      DropOutOp<GPUDevice, T>);

REGISTER_GPU(float);
//REGISTER_GPU(Eigen::half);

#undef REGISTER_GPU

#endif

} // end namespace

} // end namespace tensorflow
