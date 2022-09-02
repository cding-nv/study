#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "common_op.h"
#include "multiHeadAttention.h"

namespace tensorflow {

namespace {

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("MultiHeadAttentionGrad")
    .Input("input_q: T")
    .Input("input_k: T")
    .Input("input_v: T")
    .Input("input_w: T")
    .Input("input_dout: T")
    .Output("dq: T")
    .Output("dk: T")
    .Output("dv: T")
    .Output("dw: T")
    //.Output("out: T")
    .Attr("T: {float, half}")
    .Attr("batch: int >= 1")
    .Attr("heads: int >= 1")
    .Attr("seq_len: int >= 1")
    .Attr("hidden_size: int >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
            
            int head_num;
            int seq_len_;
            int hidden_size_;
            //c->GetAttr("batch", &batch_);
            auto batch = c->Dim(c->input(0), 0);
            c->GetAttr("heads", &head_num);
            c->GetAttr("seq_len", &seq_len_);
            c->GetAttr("hidden_size", &hidden_size_);

            if (hidden_size_ == 8) {
                c->set_output(0, c->MakeShape({batch, seq_len_, hidden_size_}));  // dq 8x4
                c->set_output(1, c->MakeShape({1, 1, 8, 10}));  // dk  
                c->set_output(2, c->MakeShape({1, 1, 8, 10}));   // dv
                c->set_output(3, c->MakeShape({1, 1, 4, 48}));  // dw
                //c->set_output(4, c->MakeShape({1, 1, 1, 32}));
            } else {  // self-attention
              c->set_output(0, c->MakeShape({batch, seq_len_ * hidden_size_}));  // dq
                c->set_output(1, c->MakeShape({batch, seq_len_ * hidden_size_}));  // dk
                c->set_output(2, c->MakeShape({batch, seq_len_ * hidden_size_}));  // dv
                c->set_output(3, c->MakeShape({4 *  hidden_size_ * hidden_size_})); // dw
                //c->set_output(4, c->MakeShape({1, 1, 1, 32}));    
            }

            return Status::OK();
    });

template <typename Device, typename T>
class MultiHeadAttentionGradOp : public CommonOp<T> {
 public:
  explicit MultiHeadAttentionGradOp(OpKernelConstruction* context): CommonOp<T>(context) {
      OP_REQUIRES_OK(context, context->GetAttr("heads", &heads_));
      OP_REQUIRES_OK(context, context->GetAttr("batch", &batch_));
      OP_REQUIRES_OK(context, context->GetAttr("seq_len", &seq_len_));
      OP_REQUIRES_OK(context, context->GetAttr("hidden_size", &hidden_size_));

      //param.cublas_handle = this->get_cublas_handler();
      //check_cuda_error(cublasSetStream(param.cublas_handle, param.stream));

      testOpts opts;
          
      // Default test parameters to be overwritten by user cmd line options.
      opts.attnTrain       = 1;
      opts.attnDataType    = CUDNN_DATA_FLOAT;
      opts.attnQueryMap    = 0;
      opts.attnNumHeads    = heads_;
      opts.attnBeamSize    = 1;
      opts.attnSmScaler    = 1.0;
      opts.attnDropoutRate = 0.0;
      opts.attnQsize       = hidden_size_;
      opts.attnKsize       = hidden_size_;
      opts.attnVsize       = hidden_size_;
      opts.attnProjQsize   = (hidden_size_ == 1024) ? (1024 / 16) : (6 / 3);
      opts.attnProjKsize   = (hidden_size_ == 1024) ? (1024 / 16) : (6 / 3);
      opts.attnProjVsize   = (hidden_size_ == 1024) ? (1024 / 16) : (6 / 3);
      opts.attnProjOsize   = (hidden_size_ == 1024) ? (1024 / 1) : (24 / 3);
      opts.attnSeqLenQ     = seq_len_;
      opts.attnSeqLenK     = (hidden_size_ == 1024) ? seq_len_:10;
      opts.attnBatchSize   = batch_;
      opts.attnDataLayout  = 0;
      opts.attnResLink     = 0;
      opts.attnSweep       = 0;
      opts.attnRandGeom    = 0;
      opts.attnRandSeed    = 1234;
      opts.attnFileDump    = 0;

      if (hidden_size_ == 1024) {
          assert(opts.attnQsize == 1024);
          assert(opts.attnProjQsize == 64);
          assert(opts.attnSeqLenQ == 384);
      } else {
          assert(opts.attnQsize == 8);
          assert(opts.attnProjQsize == 2);
          assert(opts.attnSeqLenQ == 4);
      }
      //printf("#### grad calling setup \n");
      //printf("opts.attnNumHeads = %d\n", opts.attnNumHeads);
      attnTest.setup(opts);
  }

  void Compute(OpKernelContext* context) override {
    //int rank = (int)context->input(0).dims();
    //OP_REQUIRES(context, rank >= 3,
    //            errors::InvalidArgument("Invalid rank. The rank of from tensor should be 3\
    //                                    ([batch, heads, seq_len])"));

    //std::cout << "SoftMax batch_, heads_, seq_len_ = " << batch_ << "," << heads_ << "," << seq_len_ << std::endl;
    //Softmax<T> _softmax = (typename Softmax<T>::Config(batch_, heads_, seq_len_));

    //OP_REQUIRES(context, context->num_inputs() == 1, errors::InvalidArgument("Less input arguments"));

    //param.op_context = context;
    //this->get_tensor(context, 0, &param.tensorQ);
    //this->get_tensor(context, 1, &param.tensorK);
    ///this->get_tensor(context, 2, &param.tensorV);
    //this->get_tensor(context, 3, &param.weights);
    this->get_tensor(context, 0, &attnTest.devQ);
    this->get_tensor(context, 1, &attnTest.devK);
    this->get_tensor(context, 2, &attnTest.devV);
    this->get_tensor(context, 3, &attnTest.devW);
    this->get_tensor(context, 4, &attnTest.devDO);

    Tensor * outputDQ = nullptr;
    Tensor * outputDK = nullptr;
    Tensor * outputDV = nullptr;
    Tensor * outputDW = nullptr;
    
    //printf("grad maxElemQ / maxElemK / maxElemV / maxWeights = %d/%d/%d/%d\n",
     //       attnTest.maxElemQ, attnTest.maxElemK, attnTest.maxElemV, attnTest.maxWeights);

    OP_REQUIRES_OK(
            context,
            context->allocate_output(0, {attnTest.maxElemQ}, &outputDQ));
    attnTest.devDQ = reinterpret_cast<DataType_ *>(outputDQ->flat<T>().data());

    OP_REQUIRES_OK(
            context,
            context->allocate_output(1, {attnTest.maxElemK}, &outputDK));
    attnTest.devDK = reinterpret_cast<DataType_ *>(outputDK->flat<T>().data());

    OP_REQUIRES_OK(
            context,
            context->allocate_output(2, {attnTest.maxElemV}, &outputDV));
    attnTest.devDV = reinterpret_cast<DataType_ *>(outputDV->flat<T>().data());

    OP_REQUIRES_OK(
            context,
            context->allocate_output(3, {attnTest.maxWeights/4}, &outputDW));
    attnTest.devDW = reinterpret_cast<DataType_ *>(outputDW->flat<T>().data());

//      Tensor * output = nullptr;
//      OP_REQUIRES_OK(
//              context,
//              context->allocate_output(4, {attnTest.maxElemO}, &output));
//      attnTest.devO = reinterpret_cast<DataType_ *>(output->flat<T>().data());

     check_cuda_error(cudaMalloc((void**)&attnTest.devO, attnTest.maxElemO * sizeof(float)));

    // mask is not enabled.
    //T* mask = nullptr;
    //check_cuda_error(cudaMalloc((void**)&mask, batch_ * heads_ * seq_len_ * seq_len_ * sizeof(float)));

    //check_cuda_error(cudaMemcpy(param.attr_out, param.from_tensor, batch_ * heads_ * seq_len_ * seq_len_ * sizeof(float), cudaMemcpyDeviceToDevice));

    //std::cout << "### _softmax forward........" << std::endl;
    //_softmax.Forward(batch_, (T*)param.attr_out, mask, param.stream);
    //std::cout << "### end of _softmax forward........" << std::endl;
    //printf("#### grad calling run\n");
    attnTest.run();

  }
 public:
  int heads_;
  int batch_;
  int seq_len_;
  int hidden_size_;
  typedef TFTraits<T> traits_;
  typedef typename traits_::DataType DataType_;
  //Softmax<T> _softmax;
  MultiheadAttentionTest<false, T, T> attnTest;
  cuda::MultiHeadAttentionGradParam<DataType_> param;
};

#ifdef GOOGLE_CUDA

#define REGISTER_GPU(T)                 \
    REGISTER_KERNEL_BUILDER(            \
        Name("MultiHeadAttentionGrad")      \
        .Device(DEVICE_GPU)             \
        .TypeConstraint<T>("T"),        \
      MultiHeadAttentionGradOp<GPUDevice, T>);

REGISTER_GPU(float);
//REGISTER_GPU(Eigen::half);

#undef REGISTER_GPU

#endif

} // end namespace

} // end namespace tensorflow
