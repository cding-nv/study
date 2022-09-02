#include "PositionsAndTimeDiff.h"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>


class MaskT {
private:
    int64_t _val;

public:
    MaskT(int64_t val): _val(val) {}

    __device__ int64_t
    operator()(int64_t time, int64_t mask) {
        return mask ? time : _val;
    }
};


class MaskP {
private:
    int64_t _nCol;
    int64_t _val;

public:
    MaskP(int64_t nCol, int64_t val): _nCol(nCol), _val(val) {}

    __device__ int64_t
    operator()(int64_t pos, int64_t mask) {
        return mask ? pos % _nCol + 1 : _val;
    }
};


class ComparatorD {
private:
    int64_t _nCol;
    int64_t* _data;

public:
    ComparatorD(int64_t nCol, int64_t* data): _nCol(nCol), _data(data) {}

    __device__ bool
    operator()(int64_t idxL, int64_t idxR) {
        int64_t rowL = idxL / _nCol;
        int64_t rowR = idxR / _nCol;
        if (rowL != rowR) {
            return rowL < rowR;
        }
        else {
            return _data[idxL] > _data[idxR];
        }
    }
};


class ComparatorA {
private:
    int64_t _nCol;
    int64_t* _data;

public:
    ComparatorA(int64_t nCol, int64_t* data): _nCol(nCol), _data(data) {}

    __device__ bool
    operator()(int64_t idxL, int64_t idxR) {
        int64_t rowL = idxL / _nCol;
        int64_t rowR = idxR / _nCol;
        if (rowL != rowR) {
            return rowL < rowR;
        }
        else {
            return _data[idxL] < _data[idxR];
        }
    }
};


__device__ int64_t
lookup(int64_t timediff) {
    timediff /= 1000;                       // ms -> s
    if (timediff < 60) {
        return timediff + 1;                // timediff_sec_*
    }
    else if (timediff < 3600) {
        return timediff / 60 + 60;          // timediff_min_*
    }
    else if (timediff < 86400) {
        return timediff / 3600 + 119;       // timediff_hour_*
    }
    else if (timediff < 2592000) {
        return timediff / 86400 + 142;      // timediff_day_*
    }
    else if (timediff < 31104000) {
        return timediff / 2592000 + 171;    // timediff_month_*
    }
    else if (timediff < 62208000) {
        return 183;                         // timediff_year_1
    }
    else {
        return 184;                         // timediff_year_2
    }
}


__global__ void
diffAndLookup(int nCol,
              const int64_t* __restrict__ idx_raw_ptr,
              const int64_t* __restrict__ buff_raw_ptr,
              int64_t* __restrict__ diff_raw_ptr) {
    idx_raw_ptr += blockIdx.x * nCol;
    for (int i = threadIdx.x + 1; i < nCol; i += blockDim.x) {
        int64_t idxL = idx_raw_ptr[i - 1];
        int64_t idxR = idx_raw_ptr[i];
        int64_t timediff = buff_raw_ptr[idxL] - buff_raw_ptr[idxR];
        diff_raw_ptr[idxR] = lookup(timediff);
    }
}


void
PositionsAndTimeDiff(
        cudaStream_t stream,
        const int64_t* time_raw_ptr,
        const int64_t* mask_raw_ptr,
        int64_t* pos_raw_ptr,
        int64_t* diff_raw_ptr,
        int64_t* buffer,
        int nRow,
        int nCol) {
    // 2 * time_numel
    const int time_numel = nRow * nCol;
    int64_t* idx_raw_ptr = buffer;
    int64_t* buff_raw_ptr = buffer + time_numel;

    // thrust device pointer
    thrust::device_ptr<const int64_t> time_ptr(time_raw_ptr);
    thrust::device_ptr<const int64_t> mask_ptr(mask_raw_ptr);
    thrust::device_ptr<int64_t> idx_ptr(idx_raw_ptr);
    thrust::device_ptr<int64_t> pos_ptr(pos_raw_ptr);
    thrust::device_ptr<int64_t> buff_ptr(buff_raw_ptr);
    thrust::device_ptr<int64_t> diff_ptr(diff_raw_ptr);

    // init indices
    thrust::sequence(thrust::cuda::par.on(stream), idx_ptr, idx_ptr + time_numel, 0);
    thrust::sequence(thrust::cuda::par.on(stream), pos_ptr, pos_ptr + time_numel, 0);
    // mask time
    thrust::transform(thrust::cuda::par.on(stream), time_ptr, time_ptr + time_numel, mask_ptr, buff_ptr, MaskT(0LL));
    // sort indices
    thrust::stable_sort(thrust::cuda::par.on(stream), idx_ptr, idx_ptr + time_numel, ComparatorD(nCol, buff_raw_ptr));
    // calc. adjacent difference & lookup diffid
    diffAndLookup<<<nRow, 256, 0, stream>>>(nCol, idx_raw_ptr, buff_raw_ptr, diff_raw_ptr);
    // get idx of the original time
    thrust::stable_sort(thrust::cuda::par.on(stream), pos_ptr, pos_ptr + time_numel, ComparatorA(nCol, idx_raw_ptr));
    // mask diffid
    thrust::transform(thrust::cuda::par.on(stream), diff_ptr, diff_ptr + time_numel, mask_ptr, diff_ptr, MaskT(0LL));
    // mod postion
    thrust::transform(thrust::cuda::par.on(stream), pos_ptr, pos_ptr + time_numel, mask_ptr, pos_ptr, MaskP(nCol, 0LL));
}

