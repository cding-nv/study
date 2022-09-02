#include "Swin.h"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;


__global__ void
relativePositionIndex(int64_t* index) {
    int tid = cg::this_grid().thread_rank();
    index[tid] = (blockIdx.y - threadIdx.y + blockDim.y) * (2 * blockDim.x - 1)
                + blockIdx.x - threadIdx.x - blockDim.x;
}


void
launchRelativePositionIndex(int height,
                            int width,
                            int64_t* index,
                            cudaStream_t stream = 0) {
    if (height * width > 1024) {
        errMsg(format("Unsupported window_size (%d * %d) for RelativePositionIndex", height, width));
    }
    const dim3 dimBlock(width, height);
    const dim3 dimGrid(width, height);
    relativePositionIndex<<<dimGrid, dimBlock, 0, stream>>>(index);
}


void
RelativePositionIndex::run(
        int height,
        int width,
        int64_t* index,
        cudaStream_t stream) {
    launchRelativePositionIndex(height,
                                width,
                                index,
                                stream);
}

