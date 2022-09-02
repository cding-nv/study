#include "common_op.h"

/**
 * Interface of attention grad operation
 */
namespace multiheadattention{

namespace softmaxbackprop {

template<typename T>
class SoftmaxGradParam{
 public:
   const T* grads;
   const T* softmax;
   const bool* mask;

   T* d_score;
   //T* d_tmp;
   float dropout_rate;

   //cublasHandle_t cublas_handle;
   cudaStream_t stream;
   stream_executor::Stream * tf_stream;
   OpKernelContext *op_context;

   SoftmaxGradParam(){
       grads = nullptr;
       softmax = nullptr;
       mask = nullptr;

       d_score = nullptr;
       //d_tmp = nullptr;

       stream = 0;
       tf_stream = nullptr;
       op_context = nullptr;
   }
};


template <OperationType OPType_>
class ISoftmaxGrad{
public:

    /**
     * do backward
     */
    virtual void backward() = 0;

    virtual ~ISoftmaxGrad() {}
};

template <OperationType OpType_>
class SoftmaxGrad: ISoftmaxGrad<OpType_>
{
private:
    typedef cuda::OpenMultiHeadAttentionTraits<OpType_> Traits_;
    typedef typename Traits_::DataType DataType_;
    const cudaDataType_t computeType_ = Traits_::computeType;
    const cudaDataType_t AType_ = Traits_::AType;
    const cudaDataType_t BType_ = Traits_::BType;
    const cudaDataType_t CType_ = Traits_::CType;
    const IAllocator& allocator_;
    SoftmaxGradParam<DataType_> param_;

    DataType_* buf_;
    //DataType_* grad_transpose_;  // [h, N, T_q, O/h]
    //DataType_* d_attention_;     // [h, N, T_q, T_k]
    //DataType_* d_score_;         // [h, N, T_q, T_k]

    int batch_;
    int from_seq_len_;
    int to_seq_len_;
    int head_num_;
    //int hidden_size_; // L
    int size_per_head_;

public:
    SoftmaxGrad(const IAllocator& allocator, int batch, int from_seq_len,
            int to_seq_len, int head_num, int size_per_head):
        allocator_(allocator), batch_(batch), from_seq_len_(from_seq_len), to_seq_len_(to_seq_len),
        head_num_(head_num), size_per_head_(size_per_head)
    {
        // allocate memory
        //int buf_size_0 = head_num_ * batch_ * from_seq_len_ * size_per_head_out_;
        //int buf_size_1 = head_num_ * batch_ * from_seq_len_ * to_seq_len_;

        buf_ = (DataType_ *)allocator_.malloc(sizeof(DataType_)*(
                    //buf_size_0 * 1 +
                    //buf_size_1 * 2
                    1
                     ));

        //#// grad_transpose_ --> 0
        //#grad_transpose_ = buf_;
        //#// d_attention_ --> 1
        //#d_attention_ = grad_transpose_ + buf_size_0;
        //#// d_score_ --> 1
        //#d_score_ = d_attention_ + buf_size_1; // transposed can also be used

    }

    void backward()
    {
        softmaxBack_kernelLauncher(
                param_.stream,
                param_.grads,
                param_.softmax,
                param_.mask,
                batch_,
                from_seq_len_,
                to_seq_len_,
                head_num_,
                size_per_head_
                );
    }

    void softmaxBack_kernelLauncher(
            cudaStream_t stream,
            const DataType_* grads, // input
            const DataType_* softmax, // input
            const bool* mask, // input
            const int batch,
            const int from_seq_len,
            const int to_seq_len,
            const int head_num,
            const int size_per_head
            );

    void initialize(SoftmaxGradParam<DataType_> param) {
        param_ = param;
    }

    ~SoftmaxGrad() override
    {
        allocator_.free(buf_);
    }
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
} // namespace softmaxbackprop

namespace layernormbackprop {

template<typename T>
class LayernormGradParam{
 public:
   const T* grads;
   const T* x_data;
   const T* vars;
   const T* means;
   const T* gamma;
   float alpha;

   T* d_gamma;
   T* d_betta;
   T* d_x;
   //T* d_tmp;

   //cublasHandle_t cublas_handle;
   cudaStream_t stream;
   stream_executor::Stream * tf_stream;
   OpKernelContext *op_context;

   LayernormGradParam(){
       grads = nullptr;
       x_data = nullptr;
       vars = nullptr;
       means = nullptr;
       gamma = nullptr;

       d_gamma = nullptr;
       d_betta = nullptr;
       d_x = nullptr;
       //d_tmp = nullptr;

       stream = 0;
       tf_stream = nullptr;
       op_context = nullptr;

       alpha = 1.0f;
   }
};


template <OperationType OPType_>
class ILayernormGrad{
public:

    /**
     * do backward
     */
    virtual void backward() = 0;

    virtual ~ILayernormGrad() {}
};

template <OperationType OpType_>
class LayernormGrad: ILayernormGrad<OpType_>
{
private:
    typedef cuda::OpenMultiHeadAttentionTraits<OpType_> Traits_;
    typedef typename Traits_::DataType DataType_;
    const cudaDataType_t computeType_ = Traits_::computeType;
    const cudaDataType_t AType_ = Traits_::AType;
    const cudaDataType_t BType_ = Traits_::BType;
    const cudaDataType_t CType_ = Traits_::CType;
    const IAllocator& allocator_;
    LayernormGradParam<DataType_> param_;

    DataType_ * buf_;
    DataType_* gamma_inter; //[1024, size_per_head_]
    DataType_* betta_inter; //[1024, size_per_head_]

    int batch_;
    int seq_len_;
    int head_num_;
    int hidden_size_; // L
    int size_per_head_;

public:
    LayernormGrad(const IAllocator& allocator, int batch, int seq_len,
            int head_num, int hidden_size, int size_per_head):
        allocator_(allocator), batch_(batch), seq_len_(seq_len),
        head_num_(head_num), hidden_size_(hidden_size), size_per_head_(size_per_head)
    {
        // allocate memory
        //int buf_size_0 = head_num_ * batch_ * from_seq_len_ * size_per_head_;
        int buf_size = 1024 * size_per_head_;

        buf_ = (DataType_ *)allocator_.malloc(sizeof(DataType_)*(
                    buf_size * 2
                     ));

        gamma_inter = buf_;
        betta_inter = buf_ + buf_size;
    }

    void backward()
    {
        layernormBack_kernelLauncher(
                param_.stream,
                param_.grads,
                param_.x_data,
                param_.vars,
                param_.means,
                param_.gamma,
                batch_,
                seq_len_,
                head_num_,
                hidden_size_,
                size_per_head_
                );
    }

    void layernormBack_kernelLauncher(
            cudaStream_t stream,
            const DataType_* grads, // input
            const DataType_* x_data, // input
            const DataType_* vars, // input
            const DataType_* means, // input
            const DataType_* gamma, // input
            const int batch,
            const int seq_len,
            const int head_num,
            const int hidden_size,
            const int size_per_head
            );

    void initialize(LayernormGradParam<DataType_> param) {
        param_ = param;
    }

    ~LayernormGrad() override
    {
        allocator_.free(buf_);
    }
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
} // namespace layernormbackprop

namespace densebackprop {

template<typename T>
class DenseGradParam{
 public:
   const T* q_grads;
   const T* k_grads;
   const T* v_grads;
   const T* query;
   const T* key;
   const T* value;
   const T* q_kernel;
   const T* k_kernel;
   const T* v_kernel;
   const T* query_layer; // this data is need fro backprop of leaky relu
   const T* key_layer; // this data is need fro backprop of leaky relu
   const T* value_layer; // this data is need fro backprop of leaky relu
   float alpha; // alpha in leaky relu

   T* dq;
   T* dk;
   T* dv;
   T* dwq;
   T* dbq;
   T* dwk;
   T* dbk;
   T* dwv;
   T* dbv;
   T* dv_inter;
   //T* dtmp;

   cudaStream_t stream;
   stream_executor::Stream * tf_stream;
   OpKernelContext *op_context;

   DenseGradParam(){
       q_grads = nullptr;
       k_grads = nullptr;
       v_grads = nullptr;
       query = nullptr;
       key = nullptr;
       value = nullptr;
       q_kernel = nullptr;
       k_kernel = nullptr;

       dq= nullptr;
       dk= nullptr;
       dv= nullptr;
       dwq= nullptr;
       dbq= nullptr;
       dwk= nullptr;
       dbk= nullptr;
       dv_inter = nullptr;
       //dwv= nullptr;
       dbv= nullptr;
       //dtmp=nullptr;

       alpha = 0.3f; // the default value in tensorFlow
       stream = 0;
       tf_stream = nullptr;
       op_context = nullptr;
   }
};


template <OperationType OPType_>
class IDenseGrad{
public:

    /**
     * do backward
     */
    virtual void backward() = 0;

    virtual ~IDenseGrad() {}
};

template <OperationType OpType_>
class DenseGrad: IDenseGrad<OpType_>
{
private:
    typedef cuda::OpenMultiHeadAttentionTraits<OpType_> Traits_;
    typedef typename Traits_::DataType DataType_;
    const cudaDataType_t computeType_ = Traits_::computeType;
    const cudaDataType_t AType_ = Traits_::AType;
    const cudaDataType_t BType_ = Traits_::BType;
    const cudaDataType_t CType_ = Traits_::CType;
    const IAllocator& allocator_;
    DenseGradParam<DataType_> param_;

    DataType_* buf_;
    //DataType_* dwq_inter;// [N, C_q, C]
    //DataType_* dwk_inter;// [N, C_k, C]
    DataType_* dw_inter;// [N, C_v, C]
    DataType_* grads_inter;// [N, T_k, C]

    int batch_;
    int from_seq_len_;
    int to_seq_len_;
    int head_num_;
    int hidden_size_; // L
    int size_per_head_;
    int size_per_head_out_;
    int hs_q_; // hidden_size_q_
    int hs_k_;
    int hs_v_;

public:
    DenseGrad(const IAllocator& allocator, int batch, int from_seq_len, int to_seq_len,
            int head_num, int hidden_size, int size_per_head, int size_per_head_out, int hs_q, int hs_k, int hs_v):
        allocator_(allocator), batch_(batch), from_seq_len_(from_seq_len), to_seq_len_(to_seq_len),
        head_num_(head_num), hidden_size_(hidden_size), size_per_head_(size_per_head), size_per_head_out_(size_per_head_out),
        hs_q_(hs_q), hs_k_(hs_k), hs_v_(hs_v)
    {
        // allocate memory
        int size_0 = batch_ * hs_q_ * hidden_size_;
        int size_1 = batch_ * hs_k_ * hidden_size_;
        //int buf_size_2 = batch_ * hs_v_ * head_num_ * size_per_head_out_;
        int seq_max = to_seq_len > from_seq_len ? to_seq_len : from_seq_len;
        int head_max = size_per_head_out_ > size_per_head ? size_per_head_out_ : size_per_head;
        int buf_size_3 = batch_ * seq_max * head_num_ * head_max;
        int buf_size_max = (size_0 > size_1) ? size_0 : size_1;
        //buf_size_max = (buf_size_max > buf_size_2) ? buf_size_max : buf_size_2;

        buf_ = (DataType_ *)allocator_.malloc(sizeof(DataType_)*(
                    buf_size_max +
                    buf_size_3
                     ));

        dw_inter = buf_;
        grads_inter = dw_inter + buf_size_max;
    }

    void backward()
    {
        denseBack_kernelLauncher(
                param_.stream,
                param_.q_grads,
                param_.k_grads,
                param_.v_grads,
                param_.query,
                param_.key,
                param_.value,
                param_.q_kernel,
                param_.k_kernel,
                param_.v_kernel,
                param_.query_layer,
                param_.key_layer,
                param_.value_layer,
                batch_,
                from_seq_len_,
                to_seq_len_,
                head_num_,
                hidden_size_,
                size_per_head_,
                size_per_head_out_,
                hs_q_,
                hs_k_,
                hs_v_
                );
    }

    void denseBack_kernelLauncher(
            cudaStream_t stream,
            const DataType_* q_grads, // input
            const DataType_* k_grads, // input
            const DataType_* v_grads, // input
            const DataType_* query,   // input
            const DataType_* key,     // input
            const DataType_* value,   // input
            const DataType_* q_kernel,  // input
            const DataType_* k_kernel,  // input
            const DataType_* v_kernel,  // input
            const DataType_* query_layer,  // input
            const DataType_* key_layer,  // input
            const DataType_* value_layer,  // input
            const int batch,
            const int from_seq_len,
            const int to_seq_len,
            const int head_num,
            const int hidden_size,
            const int size_per_head,
            const int size_per_head_out,
            const int hs_q,
            const int hs_k,
            const int hs_v
            );

    void initialize(DenseGradParam<DataType_> param) {
        param_ = param;
    }

    ~DenseGrad() override
    {
        allocator_.free(buf_);
    }
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
}

} // namespcae multiheadattention


// from multi_head_attention_op.h
namespace tensorflow {

namespace functor {

template <typename Device, typename T, typename DType>
struct SoftmaxGradOpFunctor {
  static void Compute(OpKernelContext* context,
                        multiheadattention::softmaxbackprop::SoftmaxGradParam<DType>& params,
                        int batch_size_,
                        int from_seq_len_,
                        int to_seq_len_,
                        int head_num_,
                        int size_per_head_
                        );
};

template <typename Device, typename T, typename DType>
struct LayernormGradOpFunctor {
  static void Compute(OpKernelContext* context,
                        multiheadattention::layernormbackprop::LayernormGradParam<DType>& params,
                        int batch_size_,
                        int seq_len_,
                        int head_num_,
                        int hidden_units_,
                        int size_per_head_
                        );
};

template <typename Device, typename T, typename DType>
struct DenseGradOpFunctor {
  static void Compute(OpKernelContext* context,
                        multiheadattention::densebackprop::DenseGradParam<DType>& params,
                        int batch_size_,
                        int from_seq_len_,
                        int to_seq_len_,
                        int head_num_,
                        int hidden_units_,
                        int size_per_head_,
                        int size_per_head_out_,
                        int hs_q_,
                        int hs_k_,
                        int hs_v_
                        );
};

} // end namespace functor

} // end namespace tensorflow

