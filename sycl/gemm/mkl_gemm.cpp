#include<iostream>
#include<string>
#include<sycl/sycl.hpp>

#include "oneapi/mkl/blas.hpp"
#include "mkl.h"

// Reference: https://spec.oneapi.io/versions/latest/elements/oneMKL/source/domains/blas/gemm.html

// using dtype = float;
// using dtype = double;
// using dtype = sycl::half;
using dtype = sycl::ext::oneapi::bfloat16;

int main(int argc, char *argv[]) {

    if (argc < 4) {
        std::cout << argv[0] << " M N K" << std::endl;
        return 0;
    }
    int M = std::stoi(argv[1]);
    int N = std::stoi(argv[2]);
    int K = std::stoi(argv[3]);
    const int lda = M, ldb = K, ldc = M;

    //dtype alpha = (dtype)1.0f, beta =(dtype)0.0f;
    float alpha = 1.0f, beta = 0.0f;

    std::vector<dtype> A(M * K, (dtype)-0.00485611);
    std::vector<dtype> B(K * N, (dtype)0.0287323);
    std::vector<dtype> C(M * N, (dtype)0.0);

    sycl::queue q;

    dtype *A_dev, *B_dev, *C_dev;
    A_dev = sycl::malloc_device<dtype>(M * K * sizeof(dtype), q);
    B_dev = sycl::malloc_device<dtype>(K * N * sizeof(dtype), q);
    C_dev = sycl::malloc_device<dtype>(M * N * sizeof(dtype), q);

    q.memcpy(A_dev, A.data(), M * K * sizeof(dtype)).wait();
    q.memcpy(B_dev, B.data(), K * N * sizeof(dtype)).wait();

    // C = alpha*A*B + beta*C
    oneapi::mkl::blas::column_major::gemm(
        q,
        oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans,
        M, N, K,
        alpha,
        A_dev, lda,
        B_dev, ldb,
        beta,
        C_dev, ldc);

    q.wait();

    q.memcpy(C.data(), C_dev, M * N * sizeof(dtype)).wait();

    for (int i = 0; i < 10; ++i) {
        std::cout << C[i] << " ";
    }
    std::cout << std::endl;

    return 0;


}

