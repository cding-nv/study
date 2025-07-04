
Reference: https://spec.oneapi.io/versions/latest/elements/oneMKL/source/domains/blas/gemm.html

## mkl_gemm.cpp
```
$ icpx -fsycl  -qmkl mkl_gemm.cpp
$ a.out 12288 1 4096
```

## dpct_gemm.cpp
```
$ icpx -fsycl  -qmkl dpct_gemm.cpp
$ a.out 12288 1 4096
```
