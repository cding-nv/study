#include "gemm.h"
#include "gemmLt.h"
#include "utils.h"
#include <chrono>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
namespace ch = std::chrono;


template <typename T>
void
test(int cnt,
     float alpha,
     int m,
     int n,
     int k,
     bool transA,
     bool transB,
     const torch::Tensor& A,
     const torch::Tensor& B) {
    ch::system_clock::time_point start, stop;
    size_t time = 0, ltTime = 0;

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    auto options = torch::TensorOptions().dtype(A.dtype()).device(A.device());
    auto C = torch::empty({m, n}, options);
    auto ltC = torch::empty_like(C);
    auto bias = torch::zeros(n, options);
    auto ws = torch::empty(1<<22, options);

    for (int i = 0; i < 4; ++i) {
        checkErr(gemmC(
                handle,
                transA,
                transB,
                m, n, k,
                (const T*)A.data_ptr(),
                (const T*)B.data_ptr(),
                (T*)C.data_ptr(),
                alpha));
    }
    checkErr(cudaDeviceSynchronize());

    for (int i = 0; i < cnt; ++i) {
        start = ch::system_clock::now();
        checkErr(gemmC(
                handle,
                transA,
                transB,
                m, n, k,
                (const T*)A.data_ptr(),
                (const T*)B.data_ptr(),
                (T*)C.data_ptr(),
                alpha));
        checkErr(cudaDeviceSynchronize());
        stop = ch::system_clock::now();
        time += ch::duration_cast<ch::nanoseconds>(stop - start).count();
    }

    for (int i = 0; i < 4; ++i) {
        gemm_bias(
                (cublasLtHandle_t)handle,
                transA,
                transB,
                m, n, k,
                (const T*)A.data_ptr(),
                (const T*)B.data_ptr(),
                (const void*)bias.data_ptr(),
                (T*)ltC.data_ptr(),
                (void*)ws.data_ptr(),
                ws.numel(),
                alpha);
    }
    checkErr(cudaDeviceSynchronize());

    for (int i = 0; i < cnt; ++i) {
        start = ch::system_clock::now();
        gemm_bias(
                (cublasLtHandle_t)handle,
                transA,
                transB,
                m, n, k,
                (const T*)A.data_ptr(),
                (const T*)B.data_ptr(),
                (const void*)bias.data_ptr(),
                (T*)ltC.data_ptr(),
                (void*)ws.data_ptr(),
                ws.numel(),
                alpha);
        checkErr(cudaDeviceSynchronize());
        stop = ch::system_clock::now();
        ltTime += ch::duration_cast<ch::nanoseconds>(stop - start).count();
    }

    bool ok = torch::equal(C, ltC);
    if (!ok) {
        int pass = (C == ltC).sum().item<int>();
        float max_abs_diff = (C - ltC).abs().max().item<float>();
        float max_rel_diff = ((C - ltC) / C).abs().max().item<float>();
        printf("  pass: %d / %d, max_abs_diff: %f, max_rel_diff: %f\n", pass, C.numel(), max_abs_diff, max_rel_diff);
    }
    printf("  %s\n", ok ? "identical" : ">>> DIFFERENT <<<");
    printf("  cublas   time: %f us\n", time / (1000. * cnt));
    printf("  cublasLt time: %f us\n", ltTime / (1000. * cnt));
}


template <typename T>
void
runTest(int cnt,
        float alpha,
        const torch::Tensor& A,
        const torch::Tensor& B,
        const torch::Tensor& AT,
        const torch::Tensor& BT) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(AT);
    CHECK_INPUT(BT);

    int m = A.size(0);
    int n = B.size(1);
    int k = A.size(1);

    printf("\n>>> NN\n");
    test<T>(cnt, alpha, m, n, k, false, false, A, B);

    printf("\n>>> TN\n");
    test<T>(cnt, alpha, m, n, k, true, false, AT, B);

    printf("\n>>> NT\n");
    test<T>(cnt, alpha, m, n, k, false, true, A, BT);

    printf("\n>>> TT\n");
    test<T>(cnt, alpha, m, n, k, true, true, AT, BT);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("runTest_fp16",
          &runTest<__half>);
    m.def("runTest_fp32",
          &runTest<float>);
}

