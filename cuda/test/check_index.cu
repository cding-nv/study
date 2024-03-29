//#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>
/*
 * Display the dimensionality of a thread block and grid from the host and
 * device.
 */
__global__ void checkIndex(void)
{
    printf("idx_x=%d idx_y=%d threadIdx:(%d, %d, %d), blockIdx:(%d, %d, %d), blockDim:(%d, %d, %d), gridDim:(%d, %d, %d)\n", 
		    threadIdx.x + blockIdx.x * blockDim.x,
                    threadIdx.y + blockIdx.y * blockDim.y,
                    threadIdx.x, threadIdx.y, threadIdx.z,
		    blockIdx.x, blockIdx.y, blockIdx.z,
		    blockDim.x, blockDim.y, blockDim.z,
		    gridDim.x, gridDim.y, gridDim.z);
    //printf("index = %d\n", threadIdx.x + blockIdx.x * blockDim.x);
    //printf("blockIdx:(%d, %d, %d)\n", blockIdx.x, blockIdx.y, blockIdx.z);
    //printf("blockDim:(%d, %d, %d)\n", blockDim.x, blockDim.y, blockDim.z);
    //printf("gridDim:(%d, %d, %d)\n", gridDim.x, gridDim.y, gridDim.z);
}
int main(int argc, char **argv)
{
    // define total data element
    int nElem = 6;
    // define grid and block structure
    dim3 block(3,3,1);
    dim3 grid(2,2,1);
    //dim3 grid((nElem + block.x - 1) / block.x);
    // check grid and block dimension from host side
    //printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
    //printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);
    // check grid and block dimension from device side
    //checkIndex<<<2, 3>>>();
    checkIndex<<<grid, block>>>();
    cudaDeviceSynchronize();

    // reset device before you leave
    //CHECK(cudaDeviceReset());
    return(0);
}
