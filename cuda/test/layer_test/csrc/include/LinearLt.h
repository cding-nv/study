#pragma once
#include "gemm.h"
#include "gemmLt.h"


// out = inp * weight^T + bias
template <typename T>
class LinearLt {
public:
    static void
    forward(const T* inp,
            const T* weight,
            const T* bias,
            T* out,
            cublasHandle_t handle,
            int64_t batchCnt,
            int64_t in_features,
            int64_t out_features,
            void* workspace,
            size_t workspaceSize) {
        cudaStream_t stream;
        cublasGetStream(handle, &stream);
        gemmC_bias(
                (cublasLtHandle_t)handle,
                stream,
                false,
                true,
                batchCnt,
                out_features,
                in_features,
                inp,
                weight,
                bias,
                out,
                workspace,
                workspaceSize);
    }

    static void
    backward(const T* grad,
             const T* inp,
             const T* weight,
             T* grad_inp,
             T* grad_weight,
             T* grad_bias,
             cublasHandle_t handle,
             int64_t batchCnt,
             int64_t in_features,
             int64_t out_features,
             void* workspace,
             size_t workspaceSize) {
        cudaStream_t stream;
        cublasGetStream(handle, &stream);
        gemmC_bgradb(
                (cublasLtHandle_t)handle,
                stream,
                true,
                false,
                out_features,
                in_features,
                batchCnt,
                grad,
                inp,
                grad_bias,
                grad_weight,
                workspace,
                workspaceSize);

        checkErr(gemmC(
                handle,
                false,
                false,
                batchCnt,
                in_features,
                out_features,
                grad,
                weight,
                grad_inp));
    }
};

