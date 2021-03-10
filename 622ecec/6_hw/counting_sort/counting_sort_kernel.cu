#include "counting_sort.h"

__global__ void counting_sort_kernel(int *input, int *sorted, int *global_hist, int num_elements)
{
	__shared__ int local_hist[MAX_VALUE - MIN_VALUE];

	int threadX = threadIdx.x;
//	int threadY = threadIdx.y;

	int blockX = blockIdx.x;
//	int blockY = blockIdx.y;

	int row = blockX*blockDim.x + threadX;

	local_hist[threadX] = 0;

	__syncthreads();

	if (row < num_elements)
		atomicAdd(&local_hist[input[row]], 1);

	__syncthreads();

	if (threadX < MAX_VALUE - MIN_VALUE) {
		atomicAdd(&global_hist[threadX], local_hist[threadX]);
	}

//	if (row > MAX_VALUE - MIN_VALUE)
//		return;

//    for (i = 1; i < range + 1; i++)
//    	histogram[i] = histogram[i - 1] + histogram[i];
//
//    int j;
//    int start_idx = 0;
//    for (i = 0; i < num_bins; i++) {
//        for (j = start_idx; j < bin[i]; j++) {
//            sorted_array[j] = i;
//        }
//        start_idx = bin[i];
//    }

    return;
}
