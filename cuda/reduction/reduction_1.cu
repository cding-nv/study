#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <numeric>
using namespace std;

__global__ void sum(int* input)
{
	const int tid = threadIdx.x;

	auto step_size = 1;
	int number_of_threads = blockDim.x;

	while (number_of_threads > 0)
	{
		if (tid < number_of_threads) // still alive?
		{
			const auto fst = tid * step_size * 2;
			const auto snd = fst + step_size;
			input[fst] += input[snd];
            printf("tid=%d, fst=%d, snd=%d, step_size=%d\n", tid, fst, snd, step_size);
		}

		step_size <<= 1; 
		number_of_threads >>= 1;
	}
}

int main()
{
	int h[] = {13, 27, 15, 14, 33, 2, 24, 6};
    const auto count = sizeof(h) / sizeof(h[0]);
    const int size = count * sizeof(int);

	int* d;
	
	cudaMalloc(&d, size);
	cudaMemcpy(d, h, size, cudaMemcpyHostToDevice);

	sum <<<1, count / 2 >>>(d);

	int result;
	cudaMemcpy(&result, d, sizeof(int), cudaMemcpyDeviceToHost);

	cout << "Sum is " << result << endl;

	getchar();

	cudaFree(d);
	//delete[] h;

	return 0;
}
