/*
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
//#include "tensorflow/contrib/rnn/kernels/blas_gemm.h" // AsDeviceMemory
#include "tensorflow/stream_executor/blas.h"

#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <curand_kernel.h>

#define LOGGING // enable logging

namespace multiheadattention
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

class Context {
public:
    Context() : _seed(42), _curr_offset(0)
    {
        curandCreateGenerator(&_gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(_gen, 123);
    }

    virtual ~Context()
    {
    }

    static Context& Instance()
    {
        static Context _ctx;
        return _ctx;
    }

    curandGenerator_t& GetRandGenerator() { return _gen; }

    std::pair<uint64_t, uint64_t> IncrementOffset(uint64_t offset_inc)
    {
        uint64_t offset = _curr_offset;
        _curr_offset += offset_inc;
        return std::pair<uint64_t, uint64_t>(_seed, offset);
    }

    void SetSeed(uint64_t new_seed) { _seed = new_seed; }

private:
    curandGenerator_t _gen;
    uint64_t _seed;
    uint64_t _curr_offset;
};


//void save_gpu_float(const float * dev_ptr, size_t size) {
//    float * h_ptr = (float *)malloc(sizeof(float) * size);
//    FILE *fp = NULL;
//
//    check_cuda_error(cudaMemcpy(h_ptr, dev_ptr, sizeof(float) *size, cudaMemcpyDeviceToHost));
//
//    fp = fopen("float_data.txt", "w+");
//    for (size_t i = 0; i < size; i++)
//        fprintf(fp, "%f  ", h_ptr[i]);
//
//    fclose(fp);
//    free(h_ptr);
//}
//
//
//void save_gpu_bool(const bool * dev_ptr, size_t size) {
//    bool * h_ptr = (bool *)malloc(sizeof(bool) * size);
//    FILE *fp = NULL;
//
//    check_cuda_error(cudaMemcpy(h_ptr, dev_ptr, sizeof(bool) *size, cudaMemcpyDeviceToHost));
//
//    fp = fopen("bool_data.txt", "w+");
//    for (size_t i = 0; i < size; i++) {
//        bool value = h_ptr[i];
//        if (value)
//            fprintf(fp, "%d  ", 1);
//        else
//            fprintf(fp, "%d  ", 0);
//    }
//
//    fclose(fp);
//    free(h_ptr);
//}

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

#ifdef GOOGLE_CUDA
using namespace tensorflow;
template <>
class Allocator<AllocatorType::TF> : public IAllocator
{
    OpKernelContext *context_;
    std::vector<Tensor> *allocated_tensor_vector;
    cudaStream_t stream_;
    void status_check(bool status) const
    {
        OP_REQUIRES(context_, status,
                errors::InvalidArgument("TF error: context->allocate_temp failed"));
    }

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

        status_check(status == tensorflow::Status::OK());

        auto flat = buf.flat<uint8>();
        void *ptr = (void *)flat.data();
        //cudaMemsetAsync(ptr, 0, buf_size, stream_);
        return ptr;
    }

    void free(void *ptr) const
    {
        return;
    }

    ~Allocator()
    {
        allocated_tensor_vector->clear();
        delete allocated_tensor_vector;
    }
};
#endif
} //namespace multiheadattention

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
    DenseWeight<T> query_weight;
    DenseWeight<T> key_weight;
    DenseWeight<T> value_weight;
    //DenseWeight<T> attention_output_weight; // commented by albert
};

template<typename T>
struct FFNWeight{
    DenseWeight<T> intermediate_weight;
    DenseWeight<T> output_weight;
};

namespace multiheadattention{

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

} // end of multiheadattention

namespace tensorflow
{
using namespace multiheadattention;
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
            //typedef __half DataType;
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
            //check_cuda_error(cublasCreate(&cublas_handle_));
        };

        template<typename DataType_>
            void get_tensor(OpKernelContext *context, int tensor_id, const DataType_** tensor_ptr, int off_set = 0){
                *tensor_ptr = reinterpret_cast<const DataType_ *>(context->input(tensor_id).flat<T>().data()) + off_set;
                OP_REQUIRES(context, *tensor_ptr != nullptr, errors::InvalidArgument("tensor %d is null", tensor_id));
            }

        //cublasHandle_t get_cublas_handler() {return cublas_handle_; }

        ~CommonOp() { /*cublasDestroy(cublas_handle_);*/ }
    private:
        //cublasHandle_t cublas_handle_;

};

} //namespace
} //namespace tensorflow

// form cuda_kernel.h
namespace multiheadattention{
    /* ********************************** common kernel *********************************** */
template <typename T>
    void add_bias_act_kernelLauncher(T* out, const T* bias, int m, int n, cudaStream_t stream);

template <typename T>
    extern void layernorm_kernelLauncher(T* out, const T* input_tensor,
            const T* gamma, const T* beta,
            int m, int n,
            cudaStream_t stream);

}//namespace multiheadattention

// from multi_head_attention.h
namespace multiheadattention{
namespace cuda{
template<typename T>
    class MultiHeadInitParam{
        public:
            const T* attention_scores;
            const bool * k_mask;
            T* softmax; // softmax
            T* output;  // the final output, output of dropout
            T * inter_output;
            const bool * mask; // dropout mask

            OpKernelContext *op_context;

            float dropout_rate;
            cudaStream_t stream;
            stream_executor::Stream * tf_stream;
            MultiHeadInitParam(){
                attention_scores = nullptr;
                softmax = nullptr;
                output = nullptr;
                mask = nullptr;
                k_mask = nullptr;
                inter_output = nullptr;

                op_context = nullptr;
                stream = 0;
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
}//namespace multiheadattention

// from open_attentin.h
namespace multiheadattention{

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


        int batch_size_;
        int from_seq_len_;
        int to_seq_len_;
        int head_num_;
        int size_per_head_;

        //cudaStream_t stream_mem_;

    public:
        //Ctor
        OpenMultiHeadAttention(const IAllocator& allocator, int batch_size, int from_seq_len,
                int to_seq_len, int head_num, int size_per_head):
            allocator_(allocator), batch_size_(batch_size), from_seq_len_(from_seq_len), to_seq_len_(to_seq_len),
            head_num_(head_num), size_per_head_(size_per_head)
    {
#ifndef NDEBUG
        PRINT_FUNC_NAME_();
#endif

        // [N, T_q, C]
        //int buf_size_0 = batch_size_ * head_num_ * from_seq_len_ * size_per_head_;
        // [M, T_k, C]
        //buf_ = (DataType_*) allocator_.malloc(sizeof(DataType_) *
        //        (buf_size_0 * 1 +
        //         buf_size_1 * 1 +
        //         buf_size_3 * 1 +
        //         buf_size_5 * 1 +
        //         qk_buf_size ) );

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

        void add_logging(std::string log, const char *const file, int const line)
        {
#ifndef NDEBUG
            cudaDeviceSynchronize();
            std::cout << log << std::endl;
            check(cudaGetLastError(), file, line);
#endif
        }

        void forward()
        {
#ifndef NDEBUG
            PRINT_FUNC_NAME_();
#endif
            float scalar = 1.0f / sqrtf(size_per_head_ * 1.0f);
            multiHeadAttr_nofuse_kernelLauncher(
                    param_.stream,
                    batch_size_,
                    from_seq_len_,
                    to_seq_len_,
                    head_num_,
                    scalar);
        }

        void multiHeadAttr_nofuse_kernelLauncher(
                cudaStream_t stream,
                const int batch_size,
                const int from_seq_len,
                const int to_seq_len,
                const int head_num,
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
            //allocator_.free(buf_);
            //cudaStreamDestroy(stream_mem_);
        }
};

}//namespace cuda
}//namespace multiheadattention

// from multi_head_attention_op.h
namespace tensorflow {

namespace functor {

template <typename Device, typename T, typename DType>
    struct MultiHeadAttentionOpFunctor {
        static void Compute(OpKernelContext* context,
                multiheadattention::cuda::MultiHeadInitParam<DType>& params,
                int batch_size_,
                int from_seq_len_,
                int to_seq_len_,
                int head_num_,
                int size_per_head_);
    };

} // end namespace functor

} // end namespace tensorflow


#endif
