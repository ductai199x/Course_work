/* Blur filter. Device code. */

#ifndef _BLUR_FILTER_KERNEL_H_
#define _BLUR_FILTER_KERNEL_H_

#include "blur_filter.h"

__global__ void 
blur_filter_kernel (const float *in, float *out, int size)
{
	/* Obtain thread index within the thread block */
	int threadX = threadIdx.x;
	int threadY = threadIdx.y;

	/* Obtain block index within the grid */
	int blockX = blockIdx.x;
	int blockY = blockIdx.y;

	/* Find position in matrix */
	int column = blockDim.x * blockX + threadX;
	int row = blockDim.y * blockY + threadY;

	if (column >= size || row >= size) {
		return;
	}

	int curr_row, curr_col;
	float blur_value = 0.0;
    int num_neighbors = 0;
//
	int i, j;
	for (i = -BLUR_SIZE; i < (BLUR_SIZE + 1); i++) {
		for (j = -BLUR_SIZE; j < (BLUR_SIZE + 1); j++) {
			/* Accumulate values of neighbors while checking for
			* boundary conditions */
			curr_row = row + i;
			curr_col = column + j;
			if ((curr_row > -1) && (curr_row < size) &&\
					(curr_col > -1) && (curr_col < size)) {
				blur_value += in[curr_row * size + curr_col];
				num_neighbors += 1;
			}
		}
	}

	out[row*size + column] = blur_value/num_neighbors;
}

#endif /* _BLUR_FILTER_KERNEL_H_ */
