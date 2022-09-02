#include "UniOps.h"
#include "errMsg.h"
#include "utils.h"
#include <cuda_fp16.h>
#include <typeinfo>


__global__ void
atomicAdd(int N,
          float* dst,
          const float* src) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        atomicAdd(dst + i, src[i]);
    }
}


__global__ void
atomicAdd(int N,
          __half* dst,
          const __half* src) {
    __half2* dst2 = reinterpret_cast<__half2*>(dst);
    const __half2* src2 = reinterpret_cast<const __half2*>(src);

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        atomicAdd(dst2 + i, src2[i]);
    }
}


// dst += src
template <typename T>
void
launchAtomicAdd(int N,
                T* dst,
                const T* src,
                cudaStream_t stream) {
    if (typeid(T) == typeid(__half)) {
        if (N & 1) {
            errMsg(format("Unsupported size (%d) for atomicAdd with FP16", N));
        }
        N >>= 1;
    }
    constexpr int dimBlock = 256;
    const int dimGrid = min(GRIDMAX, ceilDiv(N, dimBlock));

    atomicAdd<<<dimGrid, dimBlock, 0, stream>>>(N, dst, src);
}


template
void
launchAtomicAdd(int N,
                float* dst,
                const float* src,
                cudaStream_t stream);


template
void
launchAtomicAdd(int N,
                __half* dst,
                const __half* src,
                cudaStream_t stream);


template <typename T>
__global__ void
cat(int nRow,
    int nColL,
    const T* srcL,
    int nColR,
    const T* srcR,
    T* dst) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    const int nCol = nColL + nColR;
    const int N = nRow * nCol;
    const int dRow = stride / nCol;
    const int dCol = stride % nCol;

    int row = i / nCol;
    int col = i % nCol;
    for (; i < N; i += stride) {
        if (col < nColL) {
            dst[i] = srcL[row * nColL + col];
        }
        else {
            dst[i] = srcR[row * nColR + col - nColL];
        }
        row += dRow;
        col += dCol;
        if (col >= nCol) {
            col -= nCol;
            ++row;
        }
    }
}


// concatenate over dim=-1
template <typename T>
void
launchCat(int nRow,
          int nColL,
          const T* srcL,
          int nColR,
          const T* srcR,
          T* dst,
          cudaStream_t stream) {
    const int N = nRow * (nColL + nColR);
    constexpr int dimBlock = 256;
    const int dimGrid = min(GRIDMAX, ceilDiv(N, dimBlock << 2));

    cat<<<dimGrid, dimBlock, 0, stream>>>(nRow, nColL, srcL, nColR, srcR, dst);
}


template
void
launchCat(int nRow,
          int nColL,
          const float* srcL,
          int nColR,
          const float* srcR,
          float* dst,
          cudaStream_t stream);


template
void
launchCat(int nRow,
          int nColL,
          const __half* srcL,
          int nColR,
          const __half* srcR,
          __half* dst,
          cudaStream_t stream);


template <typename T>
__global__ void
copySlice(int N,
          int size1,
          int size2,
          int size3,
          T* tensor,
          const int64_t* mask,
          T val) {
    const int sliceSize = size1 * size2;
    const int stride0 = size1 * size2 * size3;
    const int stride1 = size2 * size3;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        const int64_t* currentMask = mask + (i / sliceSize << 1);
        int idxInSlice = i % sliceSize;
        int idx0 = currentMask[0];
        int idx3 = currentMask[1];
        int idx1 = idxInSlice / size2;
        int idx2 = idxInSlice % size2;
        tensor[idx0 * stride0 + idx1 * stride1 + idx2 * size3 + idx3] = val;
    }
}


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
                cudaStream_t stream) {
    const int N = size1 * size2 * len;
    constexpr int dimBlock = 256;
    const int dimGrid = min(GRIDMAX, ceilDiv(N, dimBlock << 2));

    copySlice<<<dimGrid, dimBlock, 0, stream>>>(N, size1, size2, size3, tensor, mask, T(val));
}


template
void
launchCopySlice(int size1,             // tensor.size(1)
                int size2,             // tensor.size(2)
                int size3,             // tensor.size(3)
                float* tensor,
                int len,               // mask.size(0)
                const int64_t* mask,
                float val,
                cudaStream_t stream);


template
void
launchCopySlice(int size1,             // tensor.size(1)
                int size2,             // tensor.size(2)
                int size3,             // tensor.size(3)
                __half* tensor,
                int len,               // mask.size(0)
                const int64_t* mask,
                float val,
                cudaStream_t stream);


template <typename T>
__global__ void
narrow(int nRow,
       const T* src,
       int nColL,
       T* dstL,
       int nColR,
       T* dstR) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    const int nCol = nColL + nColR;
    const int N = nRow * nCol;
    const int dRow = stride / nCol;
    const int dCol = stride % nCol;

    int row = i / nCol;
    int col = i % nCol;
    for (; i < N; i += stride) {
        if (col < nColL) {
            dstL[row * nColL + col] = src[i];
        }
        else {
            dstR[row * nColR + col - nColL] = src[i];
        }
        row += dRow;
        col += dCol;
        if (col >= nCol) {
            col -= nCol;
            ++row;
        }
    }
}


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
             cudaStream_t stream) {
    const int N = nRow * (nColL + nColR);
    constexpr int dimBlock = 256;
    const int dimGrid = min(GRIDMAX, ceilDiv(N, dimBlock << 2));

    narrow<<<dimGrid, dimBlock, 0, stream>>>(nRow, src, nColL, dstL, nColR, dstR);
}


template
void
launchNarrow(int nRow,
             const float* src,
             int nColL,
             float* dstL,
             int nColR,
             float* dstR,
             cudaStream_t stream);


template
void
launchNarrow(int nRow,
             const __half* src,
             int nColL,
             __half* dstL,
             int nColR,
             __half* dstR,
             cudaStream_t stream);


__global__ void
permuteSlice(int begin,
             int dstSize1,
             float* dst,
             int srcSize3,
             const float* src) {
    dst += ((blockIdx.y * dstSize1 + begin + blockIdx.x) * blockDim.y + threadIdx.y) * srcSize3;
    src += ((blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x + blockIdx.x) * srcSize3;

    for (int i = threadIdx.x; i < srcSize3; i += blockDim.x) {
        dst[i] = __ldg(&src[i]);
    }
}


__global__ void
permuteSlice(int begin,
             int dstSize1,
             __half* dst,
             int srcSize3,
             const __half* src) {
    __half2* dst2 = reinterpret_cast<__half2*>(dst);
    const __half2* src2 = reinterpret_cast<const __half2*>(src);

    dst2 += ((blockIdx.y * dstSize1 + begin + blockIdx.x) * blockDim.y + threadIdx.y) * srcSize3;
    src2 += ((blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x + blockIdx.x) * srcSize3;

    for (int i = threadIdx.x; i < srcSize3; i += blockDim.x) {
        dst2[i] = __ldg(&src2[i]);
    }
}


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
                   cudaStream_t stream) {
    if (typeid(T) == typeid(__half)) {
        if (srcSize3 & 1) {
            errMsg(format("Unsupported head_size (%d) for PermuteSlice with FP16", srcSize3));
        }
        srcSize3 >>= 1;
    }

    const dim3 dimBlock(min(1024 / srcSize1, 32), srcSize1);
    const dim3 dimGrid(srcSize2, srcSize0);
    permuteSlice<<<dimGrid, dimBlock, 0, stream>>>(begin, dstSize1, dst, srcSize3, src);
}


template
void
launchPermuteSlice(int begin,
                   int dstSize1,
                   float* dst,
                   int srcSize0,
                   int srcSize1,
                   int srcSize2,
                   int srcSize3,
                   const float* src,
                   cudaStream_t stream);


template
void
launchPermuteSlice(int begin,
                   int dstSize1,
                   __half* dst,
                   int srcSize0,
                   int srcSize1,
                   int srcSize2,
                   int srcSize3,
                   const __half* src,
                   cudaStream_t stream);


template <typename T>
__global__ void
sliceExpand12(int begin,
              int srcSize3,
              const T* src,
              int repeat,
              int dstSize3,
              T* dst) {
    src += blockIdx.y * srcSize3 + begin;
    dst += (blockIdx.y * gridDim.x + blockIdx.x) * repeat * dstSize3;
    extern __shared__ char _sm[];
    T* sm = reinterpret_cast<T*>(_sm);

    for (int i = threadIdx.x; i < dstSize3; i += blockDim.x) {
        sm[i] = __ldg(&src[i]);
    }
    for (int r = 0; r < repeat; ++r) {
        for (int i = threadIdx.x; i < dstSize3; i += blockDim.x) {
            dst[i] = sm[i];
        }
        dst += dstSize3;
    }
}


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
                    cudaStream_t stream) {
    const int dimBlock = min(1024, dstSize3);
    const dim3 dimGrid(dstSize2, dstSize0);
    const int smem = sizeof(T) * dstSize3;
    sliceExpand12<<<dimGrid, dimBlock, smem, stream>>>(begin, srcSize3, src, dstSize1, dstSize3, dst);
}


template
void
launchSliceExpand12(int begin,
                    int srcSize3,
                    const float* src,
                    int dstSize0,
                    int dstSize1,
                    int dstSize2,
                    int dstSize3,
                    float* dst,
                    cudaStream_t stream);


template
void
launchSliceExpand12(int begin,
                    int srcSize3,
                    const __half* src,
                    int dstSize0,
                    int dstSize1,
                    int dstSize2,
                    int dstSize3,
                    __half* dst,
                    cudaStream_t stream);


__global__ void
slicePermute(int begin,
             int srcSize1,
             const float* src,
             int dstSize3,
             float* dst) {
    dst += ((blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x + blockIdx.x) * dstSize3;
    src += ((blockIdx.y * srcSize1 + begin + blockIdx.x) * blockDim.y + threadIdx.y) * dstSize3;

    for (int i = threadIdx.x; i < dstSize3; i += blockDim.x) {
        dst[i] = __ldg(&src[i]);
    }
}


__global__ void
slicePermute(int begin,
             int srcSize1,
             const __half* src,
             int dstSize3,
             __half* dst) {
    __half2* dst2 = reinterpret_cast<__half2*>(dst);
    const __half2* src2 = reinterpret_cast<const __half2*>(src);

    dst2 += ((blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x + blockIdx.x) * dstSize3;
    src2 += ((blockIdx.y * srcSize1 + begin + blockIdx.x) * blockDim.y + threadIdx.y) * dstSize3;

    for (int i = threadIdx.x; i < dstSize3; i += blockDim.x) {
        dst2[i] = __ldg(&src2[i]);
    }
}


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
                   cudaStream_t stream) {
    if (typeid(T) == typeid(__half)) {
        if (dstSize3 & 1) {
            errMsg(format("Unsupported head_size (%d) for SlicePermute with FP16", dstSize3));
        }
        dstSize3 >>= 1;
    }

    const dim3 dimBlock(min(1024 / dstSize1, 32), dstSize1);
    const dim3 dimGrid(dstSize2, dstSize0);
    slicePermute<<<dimGrid, dimBlock, 0, stream>>>(begin, srcSize1, src, dstSize3, dst);
}


template
void
launchSlicePermute(int begin,
                   int srcSize1,
                   const float* src,
                   int dstSize0,
                   int dstSize1,
                   int dstSize2,
                   int dstSize3,
                   float* dst,
                   cudaStream_t stream);


template
void
launchSlicePermute(int begin,
                   int srcSize1,
                   const __half* src,
                   int dstSize0,
                   int dstSize1,
                   int dstSize2,
                   int dstSize3,
                   __half* dst,
                   cudaStream_t stream);


__global__ void
slicePermute3(int begin,
              int srcSize1,
              const float* srcA,
              const float* srcB,
              const float* srcC,
              int dstSize3,
              float* dstA,
              float* dstB,
              float* dstC) {
    dstA += ((blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x + blockIdx.x) * dstSize3;
    dstB += ((blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x + blockIdx.x) * dstSize3;
    dstC += ((blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x + blockIdx.x) * dstSize3;
    srcA += ((blockIdx.y * srcSize1 + begin + blockIdx.x) * blockDim.y + threadIdx.y) * dstSize3;
    srcB += ((blockIdx.y * srcSize1 + begin + blockIdx.x) * blockDim.y + threadIdx.y) * dstSize3;
    srcC += ((blockIdx.y * srcSize1 + begin + blockIdx.x) * blockDim.y + threadIdx.y) * dstSize3;

    for (int i = threadIdx.x; i < dstSize3; i += blockDim.x) {
        dstA[i] = __ldg(&srcA[i]);
        dstB[i] = __ldg(&srcB[i]);
        dstC[i] = __ldg(&srcC[i]);
    }
}


__global__ void
slicePermute3(int begin,
              int srcSize1,
              const __half* srcA,
              const __half* srcB,
              const __half* srcC,
              int dstSize3,
              __half* dstA,
              __half* dstB,
              __half* dstC) {
    __half2* dstA2 = reinterpret_cast<__half2*>(dstA);
    __half2* dstB2 = reinterpret_cast<__half2*>(dstB);
    __half2* dstC2 = reinterpret_cast<__half2*>(dstC);
    const __half2* srcA2 = reinterpret_cast<const __half2*>(srcA);
    const __half2* srcB2 = reinterpret_cast<const __half2*>(srcB);
    const __half2* srcC2 = reinterpret_cast<const __half2*>(srcC);

    dstA2 += ((blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x + blockIdx.x) * dstSize3;
    dstB2 += ((blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x + blockIdx.x) * dstSize3;
    dstC2 += ((blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x + blockIdx.x) * dstSize3;
    srcA2 += ((blockIdx.y * srcSize1 + begin + blockIdx.x) * blockDim.y + threadIdx.y) * dstSize3;
    srcB2 += ((blockIdx.y * srcSize1 + begin + blockIdx.x) * blockDim.y + threadIdx.y) * dstSize3;
    srcC2 += ((blockIdx.y * srcSize1 + begin + blockIdx.x) * blockDim.y + threadIdx.y) * dstSize3;

    for (int i = threadIdx.x; i < dstSize3; i += blockDim.x) {
        dstA2[i] = __ldg(&srcA2[i]);
        dstB2[i] = __ldg(&srcB2[i]);
        dstC2[i] = __ldg(&srcC2[i]);
    }
}


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
                    cudaStream_t stream) {
    if (typeid(T) == typeid(__half)) {
        if (dstSize3 & 1) {
            errMsg(format("Unsupported head_size (%d) for SlicePermute with FP16", dstSize3));
        }
        dstSize3 >>= 1;
    }

    const dim3 dimBlock(min(1024 / dstSize1, 32), dstSize1);
    const dim3 dimGrid(dstSize2, dstSize0);
    slicePermute3<<<dimGrid, dimBlock, 0, stream>>>(begin, srcSize1, srcA, srcB, srcC, dstSize3, dstA, dstB, dstC);
}


template
void
launchSlicePermute3(int begin,
                    int srcSize1,
                    const float* srcA,
                    const float* srcB,
                    const float* srcC,
                    int dstSize0,
                    int dstSize1,
                    int dstSize2,
                    int dstSize3,
                    float* dstA,
                    float* dstB,
                    float* dstC,
                    cudaStream_t stream);


template
void
launchSlicePermute3(int begin,
                    int srcSize1,
                    const __half* srcA,
                    const __half* srcB,
                    const __half* srcC,
                    int dstSize0,
                    int dstSize1,
                    int dstSize2,
                    int dstSize3,
                    __half* dstA,
                    __half* dstB,
                    __half* dstC,
                    cudaStream_t stream);

