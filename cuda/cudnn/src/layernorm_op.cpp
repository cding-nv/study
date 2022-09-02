#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "common_op.h"
#include "normalize_layer.h"

namespace tensorflow {

namespace {

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("LayerNorm")
    .Input("from_tensor: T")       
    .Output("output: T")
    .Attr("T: {float, half}")
    .Attr("batch: int >= 1")
    //.Attr("heads: int >= 1")
    .Attr("seq_len: int >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
            int batch_;
            int head_num;
            int seq_len_;
            c->GetAttr("batch", &batch_);
            //c->GetAttr("heads", &head_num);
            c->GetAttr("seq_len", &seq_len_);
            c->set_output(0, c->MakeShape({batch_, seq_len_}));

            return Status::OK();
    });

template <typename Device, typename T>
class LayerNormOp : public CommonOp<T> {
 public:
  explicit LayerNormOp(OpKernelConstruction* context): CommonOp<T>(context) {
      //OP_REQUIRES_OK(context, context->GetAttr("heads", &heads_));
      OP_REQUIRES_OK(context, context->GetAttr("batch", &batch_));
      OP_REQUIRES_OK(context, context->GetAttr("seq_len", &seq_len_));
      
      //self.attn_nw = nn.Parameter(torch.Tensor(self.config.hidden_size))
      //self.attn_nb = nn.Parameter(torch.Tensor(self.config.hidden_size))

      // TODO： this 2 parameters need weights and training
      check_cuda_error(cudaMalloc(&attn_layer_norm_var, batch_ * seq_len_ * sizeof(float)));
      check_cuda_error(cudaMalloc(&attn_layer_norm_mean, batch_ * seq_len_ * sizeof(float)));
  }

  void Compute(OpKernelContext* context) override {
    std::cout << "### Peak start of LayerNormOp compute" << std::endl;
    int rank = (int)context->input(0).dims();
    OP_REQUIRES(context, rank==2,
                errors::InvalidArgument("Invalid rank. The rank of from tensor should be 3\
                                        ([batch, heads, seq_len])"));

    std::cout << "batch_,  seq_len_ = " << batch_ << ","  << "," << seq_len_ << std::endl;
    // _attn_layer_norm(typename Normalize_Layer<T>::Config(batch_size,
    //                                                        seq_length,
    //                                                        hidden_size,
    //                                                        true,
    //                                                        !normalize_invertible));
    Normalize_Layer<T> _layernorm = (typename Normalize_Layer<T>::Config(batch_, 1, seq_len_, true, false));
    _layernorm.SetVar((T*)attn_layer_norm_var);
    _layernorm.SetMean((T*)attn_layer_norm_mean);

    //std::cout << "context->num_inputs() = " << context->num_inputs() << std::endl; 

    OP_REQUIRES(context, context->num_inputs() == 1, errors::InvalidArgument("Less input arguments"));

    cuda::LayernormParam<DataType_> param;
    param.cublas_handle = this->get_cublas_handler();
    check_cuda_error(cublasSetStream(param.cublas_handle, param.stream));
    param.op_context = context;
    this->get_tensor(context, 0, &param.from_tensor);

    Tensor * output = nullptr;
    OP_REQUIRES_OK(
            context,
            context->allocate_output(0, {batch_, seq_len_}, &output));
    param.attr_out = reinterpret_cast<DataType_ *>(output->flat<T>().data());

    _layernorm.Forward(4, param.attr_out, param.from_tensor, (const float*)attn_layer_norm_var, (const float*)attn_layer_norm_mean, param.stream);

    #if 1
    float* h_mem_v = reinterpret_cast<float *>(malloc(batch_  * seq_len_ * sizeof(float)));
    check_cuda_error(cudaMemcpy(h_mem_v, param.attr_out, batch_  * seq_len_ * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < batch_  * seq_len_; i++) {
        std::cout << "layernorm out ### " << h_mem_v[i] << std::endl;
    }
    float* h_mem = reinterpret_cast<float *>(malloc(seq_len_ * sizeof(float)));
    check_cuda_error(cudaMemcpy(h_mem, attn_layer_norm_var, seq_len_ * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < seq_len_; i++) {
        std::cout << "attr_attn_layer_norm_var ### " << h_mem[i] << std::endl;
    }
    check_cuda_error(cudaMemcpy(h_mem, attn_layer_norm_mean, seq_len_ * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < seq_len_; i++) {
        std::cout << "attrattn_layer_norm_mean_ ### " << h_mem[i] << std::endl;
    }
   
    #endif
  }
 private:
  int heads_;
  int batch_;
  int seq_len_;
  void* attn_layer_norm_var;
  void* attn_layer_norm_mean;
  typedef TFTraits<T> traits_;
  typedef typename traits_::DataType DataType_;
  //Softmax<T> _softmax;
};

#ifdef GOOGLE_CUDA

#define REGISTER_GPU(T)                 \
    REGISTER_KERNEL_BUILDER(            \
        Name("LayerNorm")      \
        .Device(DEVICE_GPU)             \
        .TypeConstraint<T>("T"),        \
      LayerNormOp<GPUDevice, T>);

REGISTER_GPU(float);
//REGISTER_GPU(Eigen::half);

#undef REGISTER_GPU

#endif

} // end namespace

} // end namespace tensorflow
