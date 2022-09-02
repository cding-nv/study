#include <cuda_runtime.h>
#include <cstdlib>
#include <sys/time.h>
#include <iostream>
#include <chrono>
#include <stdio.h>

__global__
//void reduction_1(void)
void reduction_1(int* g_idata, int* g_odata)
{
  /*printf("Block ID: %d SUM: %d\n", blockIdx.x, sdata[tid]);*/
  printf("Block ID: %d\n", blockIdx.x);
  //g_odata[0] = g_idata[0];
  extern __shared__ int sdata[128];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  sdata[tid] = g_idata[i];


  __syncthreads();
  printf("Peak Block ID: %d\n", blockIdx.x);

  /*for(unsigned int s = 1; s < blockDim.x; s*=2)*/
  /*{*/
    /*if (tid%(2*s)==0)*/
    /*{*/
      /*sdata[tid] += sdata[tid + s];*/
    /*}*/
    /*__syncthreads();*/
  /*}*/


  /*if (tid==0) g_odata[blockIdx.x] = sdata[0];*/

}

int main()
{
    int N = 1 << 5;
    int* g_idata;
    int* g_odata;
    cudaMallocManaged(&g_idata, N*sizeof(int));

    //Initialize input x
    int sum = 0;
    for(int i = 0; i < N; i++)
    {
      int rand_num = rand()%10;
      g_idata[i] = rand_num;
      sum += rand_num;
    }
    std::cout<<"CPU sum: "<<sum<<std::endl;
    std::cout<<g_idata[0]<<std::endl;

    //Kernel execution
    int block_size = 1024;
    int num_blocks = (N + block_size -1) / block_size;

    cudaMallocManaged(&g_odata, num_blocks*sizeof(int));

    //auto tStart = std::chrono::high_resolution_clock::now();

    reduction_1<<< 1, 1 >>>(g_idata, g_odata);
    //reduction_1<<< 2, 1 >>>();
    //std::cout<<g_odata[0]<<std::endl;
cudaDeviceSynchronize();
    //auto tEnd = std::chrono::high_resolution_clock::now();
    //float totalHost = std::chrono::duration<float, std::milli>(tEnd - tStart).count();

    int gpu_sum = 0;
    for(int i = 0 ; i < num_blocks; i++)
    {
      gpu_sum += g_odata[i];
    }

    std::cout<<"GPU sum: "<<gpu_sum<<std::endl;
    //std::cout<<"GPU time: "<<totalHost<<std::endl;


}
