#pragma once
#include "checkErr.h"
#include <cublasLt.h>
#include <cuda_fp16.h>


// C-style FP32 gemmC_bias
// >= 11.1.1
inline void
gemmC_bias(
        cublasLtHandle_t ltHandle,
        cudaStream_t stream,
        bool transA,
        bool transB,
        int64_t m,
        int64_t n,
        int64_t k,
        const float* A,
        const float* B,
        const void* bias,
        float* C,
        void* workspace,
        size_t workspaceSize = 1<<22,
        const float alpha = 1.f,
        const float beta = 0.f) {
    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;

    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    checkErr(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, transA ? m : k, transA ? k : m, transA ? m : k));
    checkErr(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, transB ? k : n, transB ? n : k, transB ? k : n));
    checkErr(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, n, m, n));

    cublasLtMatmulDesc_t operationDesc;
    checkErr(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F));
    checkErr(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opB, sizeof(opB)));
    checkErr(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opA, sizeof(opA)));
    checkErr(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    checkErr(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    cublasLtMatmulPreference_t preference;
    checkErr(cublasLtMatmulPreferenceCreate(&preference));
    checkErr(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    int returnedResults = 0;
    checkErr(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Bdesc, Adesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));
    if (returnedResults == 0) {
        checkErr(CUBLAS_STATUS_NOT_SUPPORTED);
    }

    checkErr(cublasLtMatmul(ltHandle,
                            operationDesc,
                            &alpha,
                            B,
                            Bdesc,
                            A,
                            Adesc,
                            &beta,
                            C,
                            Cdesc,
                            C,
                            Cdesc,
                            &heuristicResult.algo,
                            workspace,
                            workspaceSize,
                            stream));
    checkErr(cublasLtMatmulPreferenceDestroy(preference));
    checkErr(cublasLtMatmulDescDestroy(operationDesc));
    checkErr(cublasLtMatrixLayoutDestroy(Cdesc));
    checkErr(cublasLtMatrixLayoutDestroy(Bdesc));
    checkErr(cublasLtMatrixLayoutDestroy(Adesc));
}


// C-style FP16 gemmC_bias
// >= 11.1.1
inline void
gemmC_bias(
        cublasLtHandle_t ltHandle,
        cudaStream_t stream,
        bool transA,
        bool transB,
        int64_t m,
        int64_t n,
        int64_t k,
        const __half* A,
        const __half* B,
        const void* bias,
        __half* C,
        void* workspace,
        size_t workspaceSize = 1<<22,
        const float alpha = 1.f,
        const float beta = 0.f) {
    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;

    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    checkErr(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16F, transA ? m : k, transA ? k : m, transA ? m : k));
    checkErr(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16F, transB ? k : n, transB ? n : k, transB ? k : n));
    checkErr(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, n, m, n));

    cublasLtMatmulDesc_t operationDesc;
    checkErr(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    checkErr(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opB, sizeof(opB)));
    checkErr(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opA, sizeof(opA)));
    checkErr(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    checkErr(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    cublasLtMatmulPreference_t preference;
    checkErr(cublasLtMatmulPreferenceCreate(&preference));
    checkErr(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    int returnedResults = 0;
    checkErr(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Bdesc, Adesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));
    if (returnedResults == 0) {
        checkErr(CUBLAS_STATUS_NOT_SUPPORTED);
    }

    checkErr(cublasLtMatmul(ltHandle,
                            operationDesc,
                            &alpha,
                            B,
                            Bdesc,
                            A,
                            Adesc,
                            &beta,
                            C,
                            Cdesc,
                            C,
                            Cdesc,
                            &heuristicResult.algo,
                            workspace,
                            workspaceSize,
                            stream));
    checkErr(cublasLtMatmulPreferenceDestroy(preference));
    checkErr(cublasLtMatmulDescDestroy(operationDesc));
    checkErr(cublasLtMatrixLayoutDestroy(Cdesc));
    checkErr(cublasLtMatrixLayoutDestroy(Bdesc));
    checkErr(cublasLtMatrixLayoutDestroy(Adesc));
}


// C-style FP32 gemmC_bgradb
// >= 11.4.2
inline void
gemmC_bgradb(
        cublasLtHandle_t ltHandle,
        cudaStream_t stream,
        bool transA,
        bool transB,
        int64_t m,
        int64_t n,
        int64_t k,
        const float* A,
        const float* B,
        const void* dbias,
        float* C,
        void* workspace,
        size_t workspaceSize = 1<<22,
        const float alpha = 1.f,
        const float beta = 0.f) {
    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BGRADB;

    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    checkErr(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, transA ? m : k, transA ? k : m, transA ? m : k));
    checkErr(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, transB ? k : n, transB ? n : k, transB ? k : n));
    checkErr(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, n, m, n));

    cublasLtMatmulDesc_t operationDesc;
    checkErr(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F));
    checkErr(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opB, sizeof(opB)));
    checkErr(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opA, sizeof(opA)));
    checkErr(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &dbias, sizeof(dbias)));
    checkErr(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    cublasLtMatmulPreference_t preference;
    checkErr(cublasLtMatmulPreferenceCreate(&preference));
    checkErr(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    int returnedResults = 0;
    checkErr(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Bdesc, Adesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));
    if (returnedResults == 0) {
        checkErr(CUBLAS_STATUS_NOT_SUPPORTED);
    }

    checkErr(cublasLtMatmul(ltHandle,
                            operationDesc,
                            &alpha,
                            B,
                            Bdesc,
                            A,
                            Adesc,
                            &beta,
                            C,
                            Cdesc,
                            C,
                            Cdesc,
                            &heuristicResult.algo,
                            workspace,
                            workspaceSize,
                            stream));
    checkErr(cublasLtMatmulPreferenceDestroy(preference));
    checkErr(cublasLtMatmulDescDestroy(operationDesc));
    checkErr(cublasLtMatrixLayoutDestroy(Cdesc));
    checkErr(cublasLtMatrixLayoutDestroy(Bdesc));
    checkErr(cublasLtMatrixLayoutDestroy(Adesc));
}


// C-style FP16 gemmC_bgradb
// >= 11.4.2
inline void
gemmC_bgradb(
        cublasLtHandle_t ltHandle,
        cudaStream_t stream,
        bool transA,
        bool transB,
        int64_t m,
        int64_t n,
        int64_t k,
        const __half* A,
        const __half* B,
        const void* dbias,
        __half* C,
        void* workspace,
        size_t workspaceSize = 1<<22,
        const float alpha = 1.f,
        const float beta = 0.f) {
    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BGRADB;

    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    checkErr(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16F, transA ? m : k, transA ? k : m, transA ? m : k));
    checkErr(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16F, transB ? k : n, transB ? n : k, transB ? k : n));
    checkErr(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, n, m, n));

    cublasLtMatmulDesc_t operationDesc;
    checkErr(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    checkErr(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opB, sizeof(opB)));
    checkErr(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opA, sizeof(opA)));
    checkErr(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &dbias, sizeof(dbias)));
    checkErr(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    cublasLtMatmulPreference_t preference;
    checkErr(cublasLtMatmulPreferenceCreate(&preference));
    checkErr(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    int returnedResults = 0;
    checkErr(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Bdesc, Adesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));
    if (returnedResults == 0) {
        checkErr(CUBLAS_STATUS_NOT_SUPPORTED);
    }

    checkErr(cublasLtMatmul(ltHandle,
                            operationDesc,
                            &alpha,
                            B,
                            Bdesc,
                            A,
                            Adesc,
                            &beta,
                            C,
                            Cdesc,
                            C,
                            Cdesc,
                            &heuristicResult.algo,
                            workspace,
                            workspaceSize,
                            stream));
    checkErr(cublasLtMatmulPreferenceDestroy(preference));
    checkErr(cublasLtMatmulDescDestroy(operationDesc));
    checkErr(cublasLtMatrixLayoutDestroy(Cdesc));
    checkErr(cublasLtMatrixLayoutDestroy(Bdesc));
    checkErr(cublasLtMatrixLayoutDestroy(Adesc));
}


// C-style FP32 gemmC_gelu_bias
// >= 11.3.1
inline void
gemmC_gelu_bias(
        cublasLtHandle_t ltHandle,
        cudaStream_t stream,
        bool transA,
        bool transB,
        int64_t m,
        int64_t n,
        int64_t k,
        const float* A,
        const float* B,
        const void* bias,
        float* C,
        void* workspace,
        size_t workspaceSize = 1<<22,
        const float alpha = 1.f,
        const float beta = 0.f) {
    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_GELU_BIAS;

    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    checkErr(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, transA ? m : k, transA ? k : m, transA ? m : k));
    checkErr(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, transB ? k : n, transB ? n : k, transB ? k : n));
    checkErr(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, n, m, n));

    cublasLtMatmulDesc_t operationDesc;
    checkErr(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F));
    checkErr(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opB, sizeof(opB)));
    checkErr(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opA, sizeof(opA)));
    checkErr(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    checkErr(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    cublasLtMatmulPreference_t preference;
    checkErr(cublasLtMatmulPreferenceCreate(&preference));
    checkErr(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    int returnedResults = 0;
    checkErr(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Bdesc, Adesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));
    if (returnedResults == 0) {
        checkErr(CUBLAS_STATUS_NOT_SUPPORTED);
    }

    checkErr(cublasLtMatmul(ltHandle,
                            operationDesc,
                            &alpha,
                            B,
                            Bdesc,
                            A,
                            Adesc,
                            &beta,
                            C,
                            Cdesc,
                            C,
                            Cdesc,
                            &heuristicResult.algo,
                            workspace,
                            workspaceSize,
                            stream));
    checkErr(cublasLtMatmulPreferenceDestroy(preference));
    checkErr(cublasLtMatmulDescDestroy(operationDesc));
    checkErr(cublasLtMatrixLayoutDestroy(Cdesc));
    checkErr(cublasLtMatrixLayoutDestroy(Bdesc));
    checkErr(cublasLtMatrixLayoutDestroy(Adesc));
}


// C-style FP16 gemmC_gelu_bias
// >= 11.3.1
inline void
gemmC_gelu_bias(
        cublasLtHandle_t ltHandle,
        cudaStream_t stream,
        bool transA,
        bool transB,
        int64_t m,
        int64_t n,
        int64_t k,
        const __half* A,
        const __half* B,
        const void* bias,
        __half* C,
        void* workspace,
        size_t workspaceSize = 1<<22,
        const float alpha = 1.f,
        const float beta = 0.f) {
    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_GELU_BIAS;

    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    checkErr(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16F, transA ? m : k, transA ? k : m, transA ? m : k));
    checkErr(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16F, transB ? k : n, transB ? n : k, transB ? k : n));
    checkErr(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, n, m, n));

    cublasLtMatmulDesc_t operationDesc;
    checkErr(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    checkErr(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opB, sizeof(opB)));
    checkErr(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opA, sizeof(opA)));
    checkErr(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    checkErr(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    cublasLtMatmulPreference_t preference;
    checkErr(cublasLtMatmulPreferenceCreate(&preference));
    checkErr(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    int returnedResults = 0;
    checkErr(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Bdesc, Adesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));
    if (returnedResults == 0) {
        checkErr(CUBLAS_STATUS_NOT_SUPPORTED);
    }

    checkErr(cublasLtMatmul(ltHandle,
                            operationDesc,
                            &alpha,
                            B,
                            Bdesc,
                            A,
                            Adesc,
                            &beta,
                            C,
                            Cdesc,
                            C,
                            Cdesc,
                            &heuristicResult.algo,
                            workspace,
                            workspaceSize,
                            stream));
    checkErr(cublasLtMatmulPreferenceDestroy(preference));
    checkErr(cublasLtMatmulDescDestroy(operationDesc));
    checkErr(cublasLtMatrixLayoutDestroy(Cdesc));
    checkErr(cublasLtMatrixLayoutDestroy(Bdesc));
    checkErr(cublasLtMatrixLayoutDestroy(Adesc));
}


// C-style FP32 gemmC_gelu_aux_bias
// >= 11.3.1
inline void
gemmC_gelu_aux_bias(
        cublasLtHandle_t ltHandle,
        cudaStream_t stream,
        bool transA,
        bool transB,
        int64_t m,
        int64_t n,
        int64_t k,
        const float* A,
        const float* B,
        const void* bias,
        const void* gelu_in,
        float* C,
        void* workspace,
        size_t workspaceSize = 1<<22,
        const float alpha = 1.f,
        const float beta = 0.f) {
    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_GELU_AUX_BIAS;

    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    checkErr(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, transA ? m : k, transA ? k : m, transA ? m : k));
    checkErr(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, transB ? k : n, transB ? n : k, transB ? k : n));
    checkErr(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, n, m, n));

    cublasLtMatmulDesc_t operationDesc;
    checkErr(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F));
    checkErr(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opB, sizeof(opB)));
    checkErr(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opA, sizeof(opA)));
    checkErr(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    checkErr(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &gelu_in, sizeof(gelu_in)));
    checkErr(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &n, sizeof(n))); 
    checkErr(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    cublasLtMatmulPreference_t preference;
    checkErr(cublasLtMatmulPreferenceCreate(&preference));
    checkErr(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    int returnedResults = 0;
    checkErr(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Bdesc, Adesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));
    if (returnedResults == 0) {
        checkErr(CUBLAS_STATUS_NOT_SUPPORTED);
    }

    checkErr(cublasLtMatmul(ltHandle,
                            operationDesc,
                            &alpha,
                            B,
                            Bdesc,
                            A,
                            Adesc,
                            &beta,
                            C,
                            Cdesc,
                            C,
                            Cdesc,
                            &heuristicResult.algo,
                            workspace,
                            workspaceSize,
                            stream));
    checkErr(cublasLtMatmulPreferenceDestroy(preference));
    checkErr(cublasLtMatmulDescDestroy(operationDesc));
    checkErr(cublasLtMatrixLayoutDestroy(Cdesc));
    checkErr(cublasLtMatrixLayoutDestroy(Bdesc));
    checkErr(cublasLtMatrixLayoutDestroy(Adesc));
}


// C-style FP16 gemmC_gelu_aux_bias
// >= 11.3.1
inline void
gemmC_gelu_aux_bias(
        cublasLtHandle_t ltHandle,
        cudaStream_t stream,
        bool transA,
        bool transB,
        int64_t m,
        int64_t n,
        int64_t k,
        const __half* A,
        const __half* B,
        const void* bias,
        const void* gelu_in,
        __half* C,
        void* workspace,
        size_t workspaceSize = 1<<22,
        const float alpha = 1.f,
        const float beta = 0.f) {
    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_GELU_AUX_BIAS;

    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    checkErr(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16F, transA ? m : k, transA ? k : m, transA ? m : k));
    checkErr(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16F, transB ? k : n, transB ? n : k, transB ? k : n));
    checkErr(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, n, m, n));

    cublasLtMatmulDesc_t operationDesc;
    checkErr(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    checkErr(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opB, sizeof(opB)));
    checkErr(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opA, sizeof(opA)));
    checkErr(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    checkErr(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &gelu_in, sizeof(gelu_in)));
    checkErr(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &n, sizeof(n))); 
    checkErr(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    cublasLtMatmulPreference_t preference;
    checkErr(cublasLtMatmulPreferenceCreate(&preference));
    checkErr(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    int returnedResults = 0;
    checkErr(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Bdesc, Adesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));
    if (returnedResults == 0) {
        checkErr(CUBLAS_STATUS_NOT_SUPPORTED);
    }

    checkErr(cublasLtMatmul(ltHandle,
                            operationDesc,
                            &alpha,
                            B,
                            Bdesc,
                            A,
                            Adesc,
                            &beta,
                            C,
                            Cdesc,
                            C,
                            Cdesc,
                            &heuristicResult.algo,
                            workspace,
                            workspaceSize,
                            stream));
    checkErr(cublasLtMatmulPreferenceDestroy(preference));
    checkErr(cublasLtMatmulDescDestroy(operationDesc));
    checkErr(cublasLtMatrixLayoutDestroy(Cdesc));
    checkErr(cublasLtMatrixLayoutDestroy(Bdesc));
    checkErr(cublasLtMatrixLayoutDestroy(Adesc));
}


// C-style FP32 gemmC_dgelu_bgrad
// >= 11.3.1
inline void
gemmC_dgelu_bgrad(
        cublasLtHandle_t ltHandle,
        cudaStream_t stream,
        bool transA,
        bool transB,
        int64_t m,
        int64_t n,
        int64_t k,
        const float* A,
        const float* B,
        const void* dbias,
        const void* gelu_in,
        float* C,
        void* workspace,
        size_t workspaceSize = 1<<22,
        const float alpha = 1.f,
        const float beta = 0.f) {
    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DGELU_BGRAD;

    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    checkErr(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, transA ? m : k, transA ? k : m, transA ? m : k));
    checkErr(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, transB ? k : n, transB ? n : k, transB ? k : n));
    checkErr(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, n, m, n));

    cublasLtMatmulDesc_t operationDesc;
    checkErr(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F));
    checkErr(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opB, sizeof(opB)));
    checkErr(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opA, sizeof(opA)));
    checkErr(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &dbias, sizeof(dbias)));
    checkErr(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &gelu_in, sizeof(gelu_in)));
    checkErr(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &n, sizeof(n))); 
    checkErr(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    cublasLtMatmulPreference_t preference;
    checkErr(cublasLtMatmulPreferenceCreate(&preference));
    checkErr(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    int returnedResults = 0;
    checkErr(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Bdesc, Adesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));
    if (returnedResults == 0) {
        checkErr(CUBLAS_STATUS_NOT_SUPPORTED);
    }

    checkErr(cublasLtMatmul(ltHandle,
                            operationDesc,
                            &alpha,
                            B,
                            Bdesc,
                            A,
                            Adesc,
                            &beta,
                            C,
                            Cdesc,
                            C,
                            Cdesc,
                            &heuristicResult.algo,
                            workspace,
                            workspaceSize,
                            stream));
    checkErr(cublasLtMatmulPreferenceDestroy(preference));
    checkErr(cublasLtMatmulDescDestroy(operationDesc));
    checkErr(cublasLtMatrixLayoutDestroy(Cdesc));
    checkErr(cublasLtMatrixLayoutDestroy(Bdesc));
    checkErr(cublasLtMatrixLayoutDestroy(Adesc));
}


// C-style FP16 gemmC_dgelu_bgrad
// >= 11.3.1
inline void
gemmC_dgelu_bgrad(
        cublasLtHandle_t ltHandle,
        cudaStream_t stream,
        bool transA,
        bool transB,
        int64_t m,
        int64_t n,
        int64_t k,
        const __half* A,
        const __half* B,
        const void* dbias,
        const void* gelu_in,
        __half* C,
        void* workspace,
        size_t workspaceSize = 1<<22,
        const float alpha = 1.f,
        const float beta = 0.f) {
    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DGELU_BGRAD;

    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    checkErr(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16F, transA ? m : k, transA ? k : m, transA ? m : k));
    checkErr(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16F, transB ? k : n, transB ? n : k, transB ? k : n));
    checkErr(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, n, m, n));

    cublasLtMatmulDesc_t operationDesc;
    checkErr(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    checkErr(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opB, sizeof(opB)));
    checkErr(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opA, sizeof(opA)));
    checkErr(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &dbias, sizeof(dbias)));
    checkErr(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &gelu_in, sizeof(gelu_in)));
    checkErr(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &n, sizeof(n))); 
    checkErr(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    cublasLtMatmulPreference_t preference;
    checkErr(cublasLtMatmulPreferenceCreate(&preference));
    checkErr(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    int returnedResults = 0;
    checkErr(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Bdesc, Adesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));
    if (returnedResults == 0) {
        checkErr(CUBLAS_STATUS_NOT_SUPPORTED);
    }

    checkErr(cublasLtMatmul(ltHandle,
                            operationDesc,
                            &alpha,
                            B,
                            Bdesc,
                            A,
                            Adesc,
                            &beta,
                            C,
                            Cdesc,
                            C,
                            Cdesc,
                            &heuristicResult.algo,
                            workspace,
                            workspaceSize,
                            stream));
    checkErr(cublasLtMatmulPreferenceDestroy(preference));
    checkErr(cublasLtMatmulDescDestroy(operationDesc));
    checkErr(cublasLtMatrixLayoutDestroy(Cdesc));
    checkErr(cublasLtMatrixLayoutDestroy(Bdesc));
    checkErr(cublasLtMatrixLayoutDestroy(Adesc));
}

