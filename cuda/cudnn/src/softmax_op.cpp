#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "common_op.h"
#include "softmax.h"

namespace tensorflow {

namespace {

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("SoftMax")
    .Input("from_tensor: T")       
    .Output("output: T")
    .Attr("T: {float, half}")
    .Attr("batch: int >= 1")
    .Attr("heads: int >= 1")
    .Attr("seq_len: int >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
            int batch_;
            int head_num;
            int seq_len_;
            c->GetAttr("batch", &batch_);
            c->GetAttr("heads", &head_num);
            c->GetAttr("seq_len", &seq_len_);
            c->set_output(0, c->MakeShape({batch_, head_num, seq_len_, seq_len_}));

            return Status::OK();
    });

template <typename Device, typename T>
class SoftMaxOp : public CommonOp<T> {
 public:
  explicit SoftMaxOp(OpKernelConstruction* context): CommonOp<T>(context) {
      OP_REQUIRES_OK(context, context->GetAttr("heads", &heads_));
      OP_REQUIRES_OK(context, context->GetAttr("batch", &batch_));
      OP_REQUIRES_OK(context, context->GetAttr("seq_len", &seq_len_));

      param.cublas_handle = this->get_cublas_handler();
      check_cuda_error(cublasSetStream(param.cublas_handle, param.stream));
  }

  void Compute(OpKernelContext* context) override {
    int rank = (int)context->input(0).dims();
    OP_REQUIRES(context, rank >= 3,
                errors::InvalidArgument("Invalid rank. The rank of from tensor should be 3\
                                        ([batch, heads, seq_len])"));

    //std::cout << "SoftMax batch_, heads_, seq_len_ = " << batch_ << "," << heads_ << "," << seq_len_ << std::endl;
    Softmax<T> _softmax = (typename Softmax<T>::Config(batch_, heads_, seq_len_));

    OP_REQUIRES(context, context->num_inputs() == 1, errors::InvalidArgument("Less input arguments"));

    //param.op_context = context;
    this->get_tensor(context, 0, &param.from_tensor);

    Tensor * output = nullptr;
    OP_REQUIRES_OK(
            context,
            context->allocate_output(0, {batch_, heads_, seq_len_, seq_len_}, &output));
    param.attr_out = reinterpret_cast<DataType_ *>(output->flat<T>().data());

    // mask is not enabled.
    T* mask = nullptr;
    //check_cuda_error(cudaMalloc((void**)&mask, batch_ * heads_ * seq_len_ * seq_len_ * sizeof(float)));

    check_cuda_error(cudaMemcpy(param.attr_out, param.from_tensor, batch_ * heads_ * seq_len_ * seq_len_ * sizeof(float), cudaMemcpyDeviceToDevice));

    //std::cout << "### _softmax forward........" << std::endl;
    _softmax.Forward(batch_, (T*)param.attr_out, mask, param.stream);
    //std::cout << "### end of _softmax forward........" << std::endl;
  }
 public:
  int heads_;
  int batch_;
  int seq_len_;
  typedef TFTraits<T> traits_;
  typedef typename traits_::DataType DataType_;
  //Softmax<T> _softmax;
  cuda::SoftmaxParam<DataType_> param;
};

#ifdef GOOGLE_CUDA

#define REGISTER_GPU(T)                 \
    REGISTER_KERNEL_BUILDER(            \
        Name("SoftMax")      \
        .Device(DEVICE_GPU)             \
        .TypeConstraint<T>("T"),        \
      SoftMaxOp<GPUDevice, T>);

REGISTER_GPU(float);
//REGISTER_GPU(Eigen::half);

#undef REGISTER_GPU

#endif

} // end namespace

} // end namespace tensorflow