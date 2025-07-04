#include<iostream>
#include <dpct/dpct.hpp>
#include<sycl/sycl.hpp>
#include <dpct/blas_utils.hpp>
#include <dpct/lib_common_utils.hpp>

#include "oneapi/mkl/blas.hpp"
#include "mkl.h"
#include "sycl_kernels.h"

#include<chrono>
#include<string>


int main(int argc, char *argv[]) {

    if (argc < 4) {
        std::cout << argv[0] << " M N K" << std::endl;
        return 0;
    }
    int M = std::stoi(argv[1]);
    int N = std::stoi(argv[2]);
    int K = std::stoi(argv[3]);
    const int lda = M, ldb = K, ldc = M;
    float alpha =1.0f, beta =0.0f;

    int ite=20;

    unsigned long ASize = ite * M * K;


    std::vector<sycl::half> A(ASize, (sycl::half)1.0);
    std::vector<sycl::half> B(K * N, (sycl::half)2.0);
    std::vector<sycl::half> C(M * N, (sycl::half)0.0);

    sycl::queue q=dpct::get_in_order_queue();

    sycl::half *A_dev, *B_dev, *C_dev;
    A_dev=sycl::malloc_device<sycl::half>(ASize * sizeof(sycl::half), q);
    B_dev=sycl::malloc_device<sycl::half>(K * N * sizeof(sycl::half), q);
    C_dev=sycl::malloc_device<sycl::half>(M * N * sizeof(sycl::half), q);

    q.memcpy(A_dev, A.data(), ASize * sizeof(sycl::half)).wait();
    q.memcpy(B_dev, B.data(), K * N * sizeof(sycl::half)).wait();

    dpct::library_data_t Atype_=dpct::library_data_t::real_half;
    dpct::library_data_t Btype_=dpct::library_data_t::real_half;
    dpct::library_data_t Ctype_=dpct::library_data_t::real_half;
    dpct::library_data_t computeType_=dpct::library_data_t::real_float;

    dpct::gemm(q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, M, N, K, &alpha, A_dev, Atype_, lda, B_dev, Btype_, ldb, &beta, C_dev, Ctype_, ldc, computeType_);
    q.wait();

    auto start = std::chrono::high_resolution_clock::now();

    for(int i=0; i<ite; i++) {
        unsigned long offset = (unsigned long)i*M*K;
        dpct::gemm(q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, M, N, K, &alpha, (A_dev+offset), Atype_, lda, B_dev, Btype_, ldb, &beta, C_dev, Ctype_, ldc, computeType_);
    }
    q.wait();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout<< "M,N,K="<<M<<" ,"<<N<<" ,"<<K<<std::endl;
    std::cout << "dpct Gemm time taken: " << duration.count()*1000000/ite << " us" << std::endl;

    q.memcpy(C.data(),C_dev,M*N*sizeof(sycl::half)).wait();

    for (int i = 0; i < 10; ++i) {
        std::cout << C[i] << " ";
    }
    std::cout << std::endl;

    return 0;


}
