#pragma once
#include "checkErr.h"
#include "gemm.h"


// C = alpha * A * B
template <typename T, bool transA, bool transB>
class BatchGemm {
public:
    static void
    forward(const T* A,
            const T* B,
            T* C,
            cublasHandle_t handle,
            int batchCnt,
            int m,
            int n,
            int k,
            float alpha = 1.f) {
        checkErr(gemmC(handle,
                       transA,
                       transB,
                       batchCnt,
                       m,
                       n,
                       k,
                       A,
                       B,
                       C,
                       alpha));
    }

    static void
    forwardA(const T* A,
             const T* B,
             T* C,
             cublasHandle_t handle,
             int batchCnt,
             int m,
             int n,
             int k,
             float alpha = 1.f) {
        checkErr(gemmC(handle,
                       transA,
                       transB,
                       batchCnt,
                       m,
                       n,
                       k,
                       A,
                       B,
                       C,
                       alpha,
                       1.f));
    }

    static void
    backward(const T* grad,
             const T* A,
             const T* B,
             T* grad_A,
             T* grad_B,
             cublasHandle_t handle,
             int batchCnt,
             int m,
             int n,
             int k,
             float alpha = 1.f) {
        if (transB) {
            checkErr(gemmC(handle,
                           true,
                           transA,
                           batchCnt,
                           n,
                           k,
                           m,
                           grad,
                           A,
                           grad_B,
                           alpha));
        }
        else {
            checkErr(gemmC(handle,
                           !transA,
                           false,
                           batchCnt,
                           k,
                           n,
                           m,
                           A,
                           grad,
                           grad_B,
                           alpha));
        }

        if (transA) {
            checkErr(gemmC(handle,
                           transB,
                           true,
                           batchCnt,
                           k,
                           m,
                           n,
                           B,
                           grad,
                           grad_A,
                           alpha));
        }
        else {
            checkErr(gemmC(handle,
                           false,
                           !transB,
                           batchCnt,
                           m,
                           k,
                           n,
                           grad,
                           B,
                           grad_A,
                           alpha));
        }
    }

    static void
    backwardA(const T* grad,
              const T* A,
              const T* B,
              T* grad_A,
              T* grad_B,
              cublasHandle_t handle,
              int batchCnt,
              int m,
              int n,
              int k,
              float alpha = 1.f) {
        if (transB) {
            checkErr(gemmC(handle,
                           true,
                           transA,
                           batchCnt,
                           n,
                           k,
                           m,
                           grad,
                           A,
                           grad_B,
                           alpha));
        }
        else {
            checkErr(gemmC(handle,
                           !transA,
                           false,
                           batchCnt,
                           k,
                           n,
                           m,
                           A,
                           grad,
                           grad_B,
                           alpha));
        }

        if (transA) {
            checkErr(gemmC(handle,
                           transB,
                           true,
                           batchCnt,
                           k,
                           m,
                           n,
                           B,
                           grad,
                           grad_A,
                           alpha,
                           1.f));
        }
        else {
            checkErr(gemmC(handle,
                           false,
                           !transB,
                           batchCnt,
                           m,
                           k,
                           n,
                           grad,
                           B,
                           grad_A,
                           alpha,
                           1.f));
        }
    }

    static void
    backwardB(const T* grad,
              const T* A,
              const T* B,
              T* grad_A,
              T* grad_B,
              cublasHandle_t handle,
              int batchCnt,
              int m,
              int n,
              int k,
              float alpha = 1.f) {
        if (transB) {
            checkErr(gemmC(handle,
                           true,
                           transA,
                           batchCnt,
                           n,
                           k,
                           m,
                           grad,
                           A,
                           grad_B,
                           alpha,
                           1.f));
        }
        else {
            checkErr(gemmC(handle,
                           !transA,
                           false,
                           batchCnt,
                           k,
                           n,
                           m,
                           A,
                           grad,
                           grad_B,
                           alpha,
                           1.f));
        }

        if (transA) {
            checkErr(gemmC(handle,
                           transB,
                           true,
                           batchCnt,
                           k,
                           m,
                           n,
                           B,
                           grad,
                           grad_A,
                           alpha));
        }
        else {
            checkErr(gemmC(handle,
                           false,
                           !transB,
                           batchCnt,
                           m,
                           k,
                           n,
                           grad,
                           B,
                           grad_A,
                           alpha));
        }
    }
};

