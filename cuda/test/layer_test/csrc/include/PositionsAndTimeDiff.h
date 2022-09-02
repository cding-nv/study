#pragma once
#include <cuda_runtime.h>


void
PositionsAndTimeDiff(
        cudaStream_t stream,
        const int64_t* time_raw_ptr,
        const int64_t* mask_raw_ptr,
        int64_t* pos_raw_ptr,
        int64_t* diff_raw_ptr,
        int64_t* buffer,
        int nRow,
        int nCol);

