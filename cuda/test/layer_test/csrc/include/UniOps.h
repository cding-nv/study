#pragma once
#include <cuda_runtime.h>


// dst += src
template <typename T>
void
launchAtomicAdd(int N,
                T* dst,
                const T* src,
                cudaStream_t stream = 0);


template <typename T>
class AtomicAdd {
public:
    static void
    run(const T* src,
        T* dst,
        int N,
        cudaStream_t stream = 0) {
        launchAtomicAdd(N,
                        dst,
                        src,
                        stream);
    }
};


// concatenate over dim=-1
template <typename T>
void
launchCat(int nRow,
          int nColL,
          const T* srcL,
          int nColR,
          const T* srcR,
          T* dst,
          cudaStream_t stream = 0);


template <typename T>
class Cat {
public:
    static void
    run(int nRow,
        int nColL,
        const T* srcL,
        int nColR,
        const T* srcR,
        T* dst,
        cudaStream_t stream = 0) {
        launchCat(nRow,
                  nColL,
                  srcL,
                  nColR,
                  srcR,
                  dst,
                  stream);
    }
};


// tensor[mask[0], :, :, mask[1]] = val
template <typename T>
void
launchCopySlice(int size1,             // tensor.size(1)
                int size2,             // tensor.size(2)
                int size3,             // tensor.size(3)
                T* tensor,
                int len,               // mask.size(0)
                const int64_t* mask,
                float val,
                cudaStream_t stream = 0);


template <typename T>
class CopySlice {
public:
    static void
    run(int size1,             // tensor.size(1)
        int size2,             // tensor.size(2)
        int size3,             // tensor.size(3)
        T* tensor,
        int len,               // mask.size(0)
        const int64_t* mask,
        float val,
        cudaStream_t stream = 0) {
        launchCopySlice(size1,
                        size2,
                        size3,
                        tensor,
                        len,
                        mask,
                        val,
                        stream);
    }
};


// dstL = src.narrow(-1, 0, nColL).contiguous()
// dstR = src.narrow(-1, nColL, nColR).contiguous()
template <typename T>
void
launchNarrow(int nRow,
             const T* src,
             int nColL,
             T* dstL,
             int nColR,
             T* dstR,
             cudaStream_t stream = 0);


template <typename T>
class Narrow {
public:
    static void
    run(int nRow,
        const T* src,
        int nColL,
        T* dstL,
        int nColR,
        T* dstR,
        cudaStream_t stream = 0) {
        launchNarrow(nRow,
                     src,
                     nColL,
                     dstL,
                     nColR,
                     dstR,
                     stream);
    }
};


// dst[:, begin:end] = src.permute(0, 2, 1, 3)
template <typename T>
void
launchPermuteSlice(int begin,
                   int dstSize1,
                   T* dst,
                   int srcSize0,
                   int srcSize1,
                   int srcSize2,
                   int srcSize3,
                   const T* src,
                   cudaStream_t stream = 0);


template <typename T>
class PermuteSlice {
public:
    static void
    run(int begin,
        int dstSize1,
        T* dst,
        int srcSize0,
        int srcSize1,
        int srcSize2,
        int srcSize3,
        const T* src,
        cudaStream_t stream = 0) {
        launchPermuteSlice(begin,
                           dstSize1,
                           dst,
                           srcSize0,
                           srcSize1,
                           srcSize2,
                           srcSize3,
                           src,
                           stream);
    }
};


// dst = src[:, :, :, begin:end].expand(-1, dstSize1, dstSize2, -1).contiguous()
template <typename T>
void
launchSliceExpand12(int begin,
                    int srcSize3,
                    const T* src,
                    int dstSize0,
                    int dstSize1,
                    int dstSize2,
                    int dstSize3,
                    T* dst,
                    cudaStream_t stream = 0);


template <typename T>
class SliceExpand12 {
public:
    static void
    run(int begin,
        int srcSize3,
        const T* src,
        int dstSize0,
        int dstSize1,
        int dstSize2,
        int dstSize3,
        T* dst,
        cudaStream_t stream = 0) {
        launchSliceExpand12(begin,
                            srcSize3,
                            src,
                            dstSize0,
                            dstSize1,
                            dstSize2,
                            dstSize3,
                            dst,
                            stream);
    }
};


// dst = src[:, begin:end].permute(0, 2, 1, 3).contiguous()
template <typename T>
void
launchSlicePermute(int begin,
                   int srcSize1,
                   const T* src,
                   int dstSize0,
                   int dstSize1,
                   int dstSize2,
                   int dstSize3,
                   T* dst,
                   cudaStream_t stream = 0);


template <typename T>
class SlicePermute {
public:
    static void
    run(int begin,
        int srcSize1,
        const T* src,
        int dstSize0,
        int dstSize1,
        int dstSize2,
        int dstSize3,
        T* dst,
        cudaStream_t stream = 0) {
        launchSlicePermute(begin,
                           srcSize1,
                           src,
                           dstSize0,
                           dstSize1,
                           dstSize2,
                           dstSize3,
                           dst,
                           stream);
    }
};


// dstA = srcA[:, begin:end].permute(0, 2, 1, 3).contiguous()
// dstB = srcB[:, begin:end].permute(0, 2, 1, 3).contiguous()
// dstC = srcC[:, begin:end].permute(0, 2, 1, 3).contiguous()
template <typename T>
void
launchSlicePermute3(int begin,
                    int srcSize1,
                    const T* srcA,
                    const T* srcB,
                    const T* srcC,
                    int dstSize0,
                    int dstSize1,
                    int dstSize2,
                    int dstSize3,
                    T* dstA,
                    T* dstB,
                    T* dstC,
                    cudaStream_t stream = 0);


template <typename T>
class SlicePermute3 {
public:
    static void
    run(int begin,
        int srcSize1,
        const T* srcA,
        const T* srcB,
        const T* srcC,
        int dstSize0,
        int dstSize1,
        int dstSize2,
        int dstSize3,
        T* dstA,
        T* dstB,
        T* dstC,
        cudaStream_t stream = 0) {
        launchSlicePermute3(begin,
                            srcSize1,
                            srcA,
                            srcB,
                            srcC,
                            dstSize0,
                            dstSize1,
                            dstSize2,
                            dstSize3,
                            dstA,
                            dstB,
                            dstC,
                            stream);
    }
};

