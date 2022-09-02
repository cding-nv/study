#pragma once

#ifndef TENSORFLOW_COMMON_OP_H
#define TENSORFLOW_COMMON_OP_H

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/stream_executor/blas.h"

#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <stdexcept>

namespace fastertransformer
{
// from common.h
  enum class OperationType{FP32, FP16};
  enum class AllocatorType{CUDA, TF, TH};

#define PRINT_FUNC_NAME_() do{\
  std::cout << "[FT][CALL] " << __FUNCTION__ << " " << std::endl; \
} while (0)

static const char *_cudaGetErrorEnum(cudaError_t error) {
  return cudaGetErrorString(error);
}

static const char *_cudaGetErrorEnum(cublasStatus_t error) {
  switch (error) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";

    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";

    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "<unknown>";
}

template <typename T>
void check(T result, const char *const file, int const line) {
  if (result) {
    printf("[ERROR] CUDA runtime error: %s, in File: %s at line: %d. \n", _cudaGetErrorEnum(result), file, line);
    exit(-1);
  }
}

#define check_cuda_error(val) check((val), __FILE__, __LINE__)

/**
 * Pop current cuda device and set new device
 * i_device - device ID to set
 * o_device - device ID to pop
 * ret  - return code (the same as cudaError_t)
 */

inline cudaError_t get_set_device(int i_device, int* o_device = NULL){
    int current_dev_id = 0;
    cudaError_t err = cudaSuccess;

    if (o_device != NULL) {
        err = cudaGetDevice(&current_dev_id);
        if (err != cudaSuccess)
            return err;
        if (current_dev_id == i_device){
            *o_device = i_device;
        }
        else{
            err = cudaSetDevice(i_device);
            if (err != cudaSuccess) {
                return err;
            }
            *o_device = current_dev_id;
        }
    }
    else{
        err = cudaSetDevice(i_device);
        if (err != cudaSuccess) {
            return err;
        }
    }

    return cudaSuccess;
}

class IAllocator
{
public:
  virtual void *malloc(size_t size) const = 0;
  virtual void free(void *ptr) const = 0;
};

template <AllocatorType AllocType_>
class Allocator;

template <>
class Allocator<AllocatorType::CUDA> : public IAllocator
{
  const int device_id_;

public:
  Allocator(int device_id) : device_id_(device_id) {}

  void *malloc(size_t size) const
  {
    void *ptr = nullptr;
    int o_device = 0;
    check_cuda_error(get_set_device(device_id_, &o_device));
    check_cuda_error(cudaMalloc(&ptr, size));
    check_cuda_error(get_set_device(o_device));
    return ptr;
  }

  void free(void *ptr) const
  {
    int o_device = 0;
    check_cuda_error(get_set_device(device_id_, &o_device));
    check_cuda_error(cudaFree(ptr));
    check_cuda_error(get_set_device(o_device));
    return;
  }
};

#ifdef GOOGLE_CUDA
using namespace tensorflow;
template <>
class Allocator<AllocatorType::TF> : public IAllocator
{
  OpKernelContext *context_;
  std::vector<Tensor> *allocated_tensor_vector;
  cudaStream_t stream_;

public:
  Allocator(OpKernelContext *context, cudaStream_t stream) : context_(context), stream_(stream)
  {
    allocated_tensor_vector = new std::vector<Tensor>;
  }

  void *malloc(size_t size) const
  {
    Tensor buf;
    long long int buf_size = (long long int)size;
    tensorflow::Status status = context_->allocate_temp(DT_UINT8, TensorShape{buf_size}, &buf);
    allocated_tensor_vector->push_back(buf);

    if (status != tensorflow::Status::OK())
      throw std::runtime_error("TF error: context->allocate_temp failed");

    auto flat = buf.flat<uint8>();
    void *ptr = (void *)flat.data();
    cudaMemsetAsync(ptr, 0, buf_size, stream_);
    return ptr;
  }

  void free(void *ptr) const
  {
#ifndef NDEBUG
    printf("call from allocator free\n");
#endif
    return;
  }

  ~Allocator()
  {
    allocated_tensor_vector->clear();
    delete allocated_tensor_vector;
  }
};
#endif
} //namespace fastertransformer

// from common_structure.h
template<typename T>
struct DenseWeight{
    const T* kernel = nullptr;
    const T* bias = nullptr;
};

template<typename T>
struct LayerNormWeight{
    const T* gamma = nullptr;
    const T* beta = nullptr;
};

template<typename T>
struct AttentionWeight{
    DenseWeight<T> query_weight_0;
    DenseWeight<T> key_weight;
    DenseWeight<T> query_weight_1;
    //DenseWeight<T> attention_output_weight; // commented by albert
};

template<typename T>
struct FFNWeight{
    DenseWeight<T> intermediate_weight;
    DenseWeight<T> output_weight;
};

namespace fastertransformer{

template<OperationType OpType_>
class TransformerTraits;

template<>
class TransformerTraits<OperationType::FP32>
{
  public:
    typedef float DataType;
    static const OperationType OpType = OperationType::FP32;
    static cudaDataType_t const computeType = CUDA_R_32F;
    static cudaDataType_t const AType = CUDA_R_32F;
    static cudaDataType_t const BType = CUDA_R_32F;
    static cudaDataType_t const CType = CUDA_R_32F;
};

template<>
class TransformerTraits<OperationType::FP16>
{
  public:
    typedef Eigen::half DataType;
    static const OperationType OpType = OperationType::FP16;
    static cudaDataType_t const computeType = CUDA_R_16F;
    static cudaDataType_t const AType = CUDA_R_16F;
    static cudaDataType_t const BType = CUDA_R_16F;
    static cudaDataType_t const CType = CUDA_R_16F;
};

} // end of fastertransformer


namespace tensorflow
{

    using namespace fastertransformer;

    template <typename T> class TFTraits;

    template <>
        class TFTraits<float>
        {
            public:
                typedef float DataType;
                static const OperationType OpType = OperationType::FP32;
        };

    template <>
        class TFTraits<Eigen::half>
        {
            public:
                typedef Eigen::half DataType;
                static const OperationType OpType = OperationType::FP16;
        };

namespace
{
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

template <typename T>
class CommonOp : public OpKernel
{
public:
  explicit CommonOp(OpKernelConstruction *context) : OpKernel(context) {
      check_cuda_error(cublasCreate(&cublas_handle_));
  };

  template<typename DataType_>
  void get_tensor(OpKernelContext *context, int tensor_id, const DataType_** tensor_ptr, int off_set = 0){
    *tensor_ptr = reinterpret_cast<const DataType_ *>(context->input(tensor_id).flat<T>().data()) + off_set;
    OP_REQUIRES(context, *tensor_ptr != nullptr, errors::InvalidArgument("tensor %d is null", tensor_id));
  }

  cublasHandle_t get_cublas_handler() {return cublas_handle_; }

  ~CommonOp() { cublasDestroy(cublas_handle_); }
private:
  cublasHandle_t cublas_handle_;

};

} //namespace
} //namespace tensorflow



// form cuda_kernel.h
namespace fastertransformer{

/* ********************************** common kernel *********************************** */

template <typename T>
void add_bias_act_kernelLauncher(T* out, const T* bias, int m, int n, cudaStream_t stream);

template <typename T>
extern void layernorm_kernelLauncher(T* out, const T* input_tensor,
                              const T* gamma, const T* beta,
                              int m, int n,
                              cudaStream_t stream);

}//namespace fastertransformer

// from multi_head_attention.h
namespace fastertransformer{
namespace cuda{

template<typename T>
class MultiHeadInitParam{
 public:
   const T* from_tensor;
   const T* to_tensor;
   const int* indices;
   AttentionWeight<T> self_attention;
   T* attr_out;
   T* inter_out; // alphas: the result of softmax operation
   T* query_1;
   T* query_2;
   T* key_1;

   const int* sequence_id_offset;
   int valid_word_num;
   cublasHandle_t cublas_handle;
   cudaStream_t stream;
   stream_executor::Stream * tf_stream;
   OpKernelContext *op_context;

   MultiHeadInitParam(){
     from_tensor = nullptr;
     to_tensor = nullptr;
     indices = nullptr;
     attr_out = nullptr;
     inter_out = nullptr;
     query_1 = nullptr;
     query_2 = nullptr;
     key_1 = nullptr;
     cublas_handle = nullptr;
     sequence_id_offset = nullptr;
     stream = 0;
     tf_stream = nullptr;
     op_context = nullptr;
   }
};

template<typename T>
class FeedForwardGradParam{
 public:
   const T* grad_key;
   const T* key;
   const T* w0;
   const T* b0;
   const int* indices;
   //AttentionWeight<T> self_attention;
   T* d_key;
   T* d_w0;
   T* d_b0;
   //T* inter;

   cublasHandle_t cublas_handle;
   cudaStream_t stream;
   stream_executor::Stream * tf_stream;
   OpKernelContext *op_context;

   FeedForwardGradParam(){
     grad_key = nullptr;
     key = nullptr;
     w0 = nullptr;
     b0 = nullptr;
     indices = nullptr;
     d_key = nullptr;
     d_w0 = nullptr;
     d_b0 = nullptr;
     //inter = nullptr;
     cublas_handle = nullptr;
     stream = 0;
      tf_stream = nullptr;
      op_context = nullptr;
   }
};

template<typename T>
class SoftmaxParam{
 public:
   const T* from_tensor;
   T* attr_out;
   cublasHandle_t cublas_handle;
   cudaStream_t stream;
   stream_executor::Stream * tf_stream;
   OpKernelContext *op_context;

   SoftmaxParam(){
     from_tensor = nullptr;
     attr_out = nullptr;
     cublas_handle = nullptr;
     stream = 0;
      tf_stream = nullptr;
      op_context = nullptr;
   }
};

template<typename T>
class SoftmaxGradParam{
 public:
   const T* grad;
   const T* softmax_out;
   T* attr_out;
   cublasHandle_t cublas_handle;
   cudaStream_t stream;
   stream_executor::Stream * tf_stream;
   OpKernelContext *op_context;

   SoftmaxGradParam(){
     grad = nullptr;
     softmax_out;
     attr_out = nullptr;
     cublas_handle = nullptr;
     stream = 0;
      tf_stream = nullptr;
      op_context = nullptr;
   }
};

template<typename T>
class DropoutParam{
 public:
   const T* from_tensor;
   T* attr_out;
   cublasHandle_t cublas_handle;
   cudaStream_t stream;
   stream_executor::Stream * tf_stream;
   OpKernelContext *op_context;

   DropoutParam(){
     from_tensor = nullptr;
     attr_out = nullptr;
     cublas_handle = nullptr;
     stream = 0;
     tf_stream = nullptr;
     op_context = nullptr;
   }
};

template<typename T>
class LayernormParam{
 public:
   const T* from_tensor;
   T* attr_out;
   cublasHandle_t cublas_handle;
   cudaStream_t stream;
   stream_executor::Stream * tf_stream;
   OpKernelContext *op_context;

   LayernormParam(){
     from_tensor = nullptr;
     attr_out = nullptr;
     cublas_handle = nullptr;
     stream = 0;
     tf_stream = nullptr;
     op_context = nullptr;
   }
};

template<typename T>
class MultiHeadAttentionParam{
 public:
   const T* tensorQ;
   const T* tensorK;
   const T* tensorV;
   const T* weights;
   T* attr_out;
   cublasHandle_t cublas_handle;
   cudaStream_t stream;
   stream_executor::Stream * tf_stream;
   OpKernelContext *op_context;

   MultiHeadAttentionParam(){
     tensorQ = nullptr;
     tensorK = nullptr;
     tensorV = nullptr;
     weights = nullptr;
     attr_out = nullptr;
     cublas_handle = nullptr;
     stream = 0;
      tf_stream = nullptr;
      op_context = nullptr;
   }
};

template<typename T>
class MultiHeadAttentionGradParam{
 public:
   const T* tensorQ;
   const T* tensorK;
   const T* tensorV;
   const T* weights;
   T* attr_out;
   cublasHandle_t cublas_handle;
   cudaStream_t stream;
   stream_executor::Stream * tf_stream;
   OpKernelContext *op_context;

   MultiHeadAttentionGradParam(){
     tensorQ = nullptr;
     tensorK = nullptr;
     tensorV = nullptr;
     weights = nullptr;
     attr_out = nullptr;
     cublas_handle = nullptr;
     stream = 0;
      tf_stream = nullptr;
      op_context = nullptr;
   }
};

/**
 * Interface of attention operation
 */
template<OperationType OpType_>
class IMultiHeadAttention{
 public:
//  typedef MultiHeadInitParam<OpType_> InitParam;
  /**
   * do forward
   **/
  virtual void forward() = 0;

  /**
   * Initialize the parameters in class
   * We will keep the Ctor empty to ensure the sub classes follow the same init routine.
   * Please be aware that no dynamic memory allocation should be placed
   **/
//  virtual void free() = 0;

  virtual ~IMultiHeadAttention(){}

};


}//namespace cuda
}//namespace fastertransformer

// from open_attentin.h
namespace fastertransformer{

template <typename T>
void add_bias_act_kernelLauncher(T* out, const T* bias, int m, int n, cudaStream_t stream);

template <typename T>
extern void layernorm_kernelLauncher(T* out, const T* input_tensor,
                              const T* gamma, const T* beta,
                              int m, int n,
                              cudaStream_t stream);

namespace cuda{

template<OperationType OpType_>
class OpenMultiHeadAttentionTraits;

template<>
class OpenMultiHeadAttentionTraits<OperationType::FP32>
{
 public:
  typedef float DataType;
  static cudaDataType_t const computeType = CUDA_R_32F;
  static cudaDataType_t const AType = CUDA_R_32F;
  static cudaDataType_t const BType = CUDA_R_32F;
  static cudaDataType_t const CType = CUDA_R_32F;
  //others
};

template<>
class OpenMultiHeadAttentionTraits<OperationType::FP16>
{
 public:
  typedef Eigen::half DataType;
  static cudaDataType_t const computeType = CUDA_R_16F;
  static cudaDataType_t const AType = CUDA_R_16F;
  static cudaDataType_t const BType = CUDA_R_16F;
  static cudaDataType_t const CType = CUDA_R_16F;
  //others
};

using namespace stream_executor;

template <typename T>
    DeviceMemory<T> ToDeviceMemory(const T * cuda_memory, uint64_t size) {
        DeviceMemoryBase wrapped(const_cast<T *>(cuda_memory), size * sizeof(T));
        DeviceMemory<T> typed(wrapped);
        return typed;
    }

template <typename T>
    DeviceMemory<T> ToDeviceMemory(const T * cuda_memory) {
        DeviceMemoryBase wrapped(const_cast<T *>(cuda_memory));
        DeviceMemory<T> typed(wrapped);
        return typed;
    }

/**
 * Multi-head attetion open sourced
 */
template<OperationType OpType_>
class OpenMultiHeadAttention: IMultiHeadAttention<OpType_>
{
 private:
  typedef OpenMultiHeadAttentionTraits<OpType_> Traits_;
  typedef typename Traits_::DataType DataType_;
  const cudaDataType_t computeType_ = Traits_::computeType;
  const cudaDataType_t AType_ = Traits_::AType;
  const cudaDataType_t BType_ = Traits_::BType;
  const cudaDataType_t CType_ = Traits_::CType;
  const IAllocator& allocator_;
  MultiHeadInitParam<DataType_> param_;

  int cublasAlgo_[4];

  DataType_* buf_;
  DataType_* q_buf_0_;
  DataType_* q_buf_1_;
  DataType_* query_buf_0_;
  DataType_* query_buf_1_;
  DataType_* k_buf_;
  DataType_* key_buf_;
  DataType_* qk_buf_;
  DataType_* transpose_dst_;
  DataType_* out_sign_; // [N, T_k]

  int batch_n_;
  int batch_m_;
  int from_seq_len_;
  int to_seq_len_;
  int head_num_;
  int hidden_size_; // L
  //int size_per_head_;
  int D_Q_0_;
  int D_Q_1_;
  int D_K_0_;

 public:
  //Ctor
  OpenMultiHeadAttention(const IAllocator& allocator, int batch_n, int batch_m, int from_seq_len,
      int to_seq_len, int head_num, int D_Q_0, int D_Q_1, int D_K_0, int hidden_size):
    allocator_(allocator), batch_n_(batch_n), batch_m_(batch_m), from_seq_len_(from_seq_len), to_seq_len_(to_seq_len),
    head_num_(head_num), D_Q_0_(D_Q_0), D_Q_1_(D_Q_1), D_K_0_(D_K_0), hidden_size_(hidden_size)
   {
#ifndef NDEBUG
    PRINT_FUNC_NAME_();
#endif

    int buf_size_0 = batch_n_ * head_num_ * from_seq_len_ * D_Q_0_; // q_buf_0_
    int buf_size_1 = batch_n_ * head_num_ * from_seq_len_ * D_Q_1_; //q_buf_1_
    int buf_size_2 = batch_n_ * head_num_ * to_seq_len_ * D_K_0_; //k_buf_
    int buf_size_3 =  batch_n_ * head_num_ * to_seq_len_ * from_seq_len_; //qk_buf_
    int buf_size_4 = batch_n_ * head_num_ * to_seq_len_ * D_K_0_; //transpose_dst_

    int buf_size_5 = batch_m_ * head_num_ * to_seq_len_ * D_K_0_; //k_buf_


    buf_ = (DataType_*) allocator_.malloc(sizeof(DataType_) *
            (buf_size_0 * 2 +
            buf_size_1 * 2 +
            buf_size_2 * 1 +
            buf_size_3 +
            buf_size_4 +
            buf_size_5));

    // [N, T_q, DQ0*h] --> 0
    q_buf_0_ = buf_;

    // (h*N, T_q, DQ0) --> 0
    query_buf_0_ = buf_ + buf_size_0;

    // [h*N, T_q, DQ1] --> 1
    q_buf_1_ = query_buf_0_ + buf_size_0;

    // [h*N, T_q, DQ1] --> 1
    query_buf_1_ = q_buf_1_ + buf_size_1;

    // [M, T_k, DK0*h] --> 2 --> 5
    k_buf_ = query_buf_1_ + buf_size_1;

    // [h*N, T_k, DK0] --> 2
    key_buf_ = k_buf_ + buf_size_2;

    // [h*N, T_k, T_q] --> 3
    qk_buf_ = key_buf_ + buf_size_2;

    // [h*N, T_k, DK0] --> 4
    transpose_dst_ = qk_buf_ + buf_size_3;

    if(OpType_ == OperationType::FP32)
    {
        cublasAlgo_[0] = -1;
        cublasAlgo_[1] = -1;
        cublasAlgo_[2] = -1;
        cublasAlgo_[3] = -1;
    }
    else
    {
        cublasAlgo_[0] = 99;
        cublasAlgo_[1] = 99;
        cublasAlgo_[2] = 99;
        cublasAlgo_[3] = 99;
    }
  }

  void forward()
  {
#ifndef NDEBUG
    PRINT_FUNC_NAME_();
#endif
    int m = param_.sequence_id_offset == nullptr ? batch_n_ * from_seq_len_ : param_.valid_word_num;
    int k = hidden_size_;
    int n = head_num_ * D_Q_0_;

    //const DataType_ alpha = (DataType_)1.0f, beta = (DataType_)0.0f;

//    check_cuda_error(cublasGemmEx(param_.cublas_handle,
//                CUBLAS_OP_N, CUBLAS_OP_N,
//                n, m, k,
//                &alpha,
//                param_.self_attention.query_weight_0.kernel, AType_, n,
//                param_.from_tensor, BType_, k,
//                &beta,
//                q_buf_0_, CType_, n,
//                computeType_,
//                static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));

    float alpha = 1.0f, beta = 0.0f;
    bool blas_launch_status = false;
    auto a_ptr = ToDeviceMemory(param_.self_attention.query_weight_0.kernel);
    auto b_ptr = ToDeviceMemory(param_.from_tensor);
    auto c_ptr = ToDeviceMemory(q_buf_0_);

    blas_launch_status = param_.tf_stream
        ->ThenBlasGemm(
                blas::Transpose::kNoTranspose,
                blas::Transpose::kNoTranspose,
                (uint64)n, (uint64)m, (uint64)k,
                alpha,
                a_ptr, n,
                b_ptr, k,
                beta,
                &c_ptr, n
                ).ok();

    OP_REQUIRES(param_.op_context, blas_launch_status, errors::InvalidArgument("ERROR while calling ThenBlasGemm.\n"));


#ifndef NDEBUG
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
#endif

    m = batch_m_ * to_seq_len_;
    k = hidden_size_;
    n = D_K_0_ * head_num_;
//    check_cuda_error(cublasGemmEx(param_.cublas_handle,
//                CUBLAS_OP_N, CUBLAS_OP_N,
//                n, m, k,
//                &alpha,
//                param_.self_attention.key_weight.kernel, AType_, n,
//                param_.to_tensor, BType_, k,
//                &beta,
//                k_buf_, CType_, n,
//                computeType_,
//                static_cast<cublasGemmAlgo_t>(cublasAlgo_[0])));

    a_ptr = ToDeviceMemory(param_.self_attention.key_weight.kernel);
    b_ptr = ToDeviceMemory(param_.to_tensor);
    c_ptr = ToDeviceMemory(k_buf_);
    blas_launch_status = param_.tf_stream
        ->ThenBlasGemm(
                blas::Transpose::kNoTranspose,
                blas::Transpose::kNoTranspose,
                (uint64)n, (uint64)m, (uint64)k,
                alpha,
                a_ptr, n,
                b_ptr, k,
                beta,
                &c_ptr, n
                ).ok();

    OP_REQUIRES(param_.op_context, blas_launch_status, errors::InvalidArgument("ERROR while calling ThenBlasGemm.\n"));
#ifndef NDEBUG
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
#endif

    //DataType_ scalar = 1 / sqrtf(size_per_head_ * 1.0f);
    float  scalar = 1.0f;
    multiHeadAttr_nofuse_kernelLauncher(
      param_.stream,
      param_.cublas_handle,
      q_buf_0_,
      param_.self_attention.query_weight_0.bias,
      k_buf_,
      param_.self_attention.key_weight.bias,
      q_buf_1_,
      param_.self_attention.query_weight_1.bias,
      param_.attr_out,
      param_.indices,
      batch_n_,
      batch_m_,
      from_seq_len_,
      to_seq_len_,
      head_num_,
      D_Q_0_,
      D_Q_1_,
      D_K_0_,
      scalar);
  }

  void multiHeadAttr_nofuse_kernelLauncher(
      cudaStream_t stream,
      cublasHandle_t handle,
      DataType_* q_buf_0,
      const DataType_* bias_Q_0,
      DataType_* k_buf,
      const DataType_* bias_K,
      DataType_* q_buf_1,
      const DataType_* bias_Q_1,
      DataType_* dst,
      const int* indices,
      const int batch_n,
      const int batch_m,
      const int from_seq_len,
      const int to_seq_len,
      const int head_num,
      const int D_Q_0,
      const int D_Q_1,
      const int D_K_0,
      const float scalar);

  void initialize(MultiHeadInitParam<DataType_> param)
  {
#ifndef NDEBUG
    PRINT_FUNC_NAME_();
#endif
    param_ = param;
  }

  ~OpenMultiHeadAttention() override
  {
    allocator_.free(buf_);
  }
};

}//namespace cuda
}//namespace fastertransformer

// from multi_head_attention_op.h
namespace tensorflow {

namespace functor {

template <typename Device, typename T, typename DType>
struct MultiHeadAttentionOpFunctor {
  static void Compute(OpKernelContext* context,
                        fastertransformer::cuda::MultiHeadInitParam<DType>& params,
                        int batch_n_,
                        int batch_m_,
                        int from_seq_len_,
                        int to_seq_len_,
                        int head_num_,
                        int D_Q_0_,
                        int D_Q_1_,
                        int D_K_0_,
                        int hidden_size_
                        );
};

} // end namespace functor

} // end namespace tensorflow

//void print_to_file_float(float* result, const int size, char* file)
//{
//  FILE* fd = fopen(file, "w");
//  float* tmp = (float*)malloc(sizeof(float) * size);
//  cudaMemcpy(tmp, result, sizeof(float) * size, cudaMemcpyDeviceToHost);
//  for(int i = 0; i < size; ++i)
//    fprintf(fd, "%f\n", tmp[i]);
//  free(tmp);
//  fclose(fd);
//}
//

#endif
