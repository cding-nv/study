#pragma once
#include "checkErr.h"
#include "gemm.h"


// out = inp * weight^T
// deal with bias in other functions
template <typename T>
class Linear {
public:
    static void
    forward(const T* inp,
            const T* weight,
            T* out,
            cublasHandle_t handle,
            int batchCnt,
            int in_features,
            int out_features) {
        checkErr(gemmC(handle,
                       false,
                       true,
                       batchCnt,
                       out_features,
                       in_features,
                       inp,
                       weight,
                       out));
    }

    static void
    backward(const T* grad,
             const T* inp,
             const T* weight,
             T* grad_inp,
             T* grad_weight,
             cublasHandle_t handle,
             int batchCnt,
             int in_features,
             int out_features) {
        checkErr(gemmC(handle,
                       true,
                       false,
                       out_features,
                       in_features,
                       batchCnt,
                       grad,
                       inp,
                       grad_weight));

        checkErr(gemmC(handle,
                       false,
                       false,
                       batchCnt,
                       in_features,
                       out_features,
                       grad,
                       weight,
                       grad_inp));
    }

    static void
    backwardA(const T* grad,
              const T* inp,
              const T* weight,
              T* grad_inp,
              T* grad_weight,
              cublasHandle_t handle,
              int batchCnt,
              int in_features,
              int out_features) {
        checkErr(gemmC(handle,
                       true,
                       false,
                       out_features,
                       in_features,
                       batchCnt,
                       grad,
                       inp,
                       grad_weight));

        checkErr(gemmC(handle,
                       false,
                       false,
                       batchCnt,
                       in_features,
                       out_features,
                       grad,
                       weight,
                       grad_inp,
                       1.f,
                       1.f));
    }
};

