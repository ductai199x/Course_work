#include "seperable_convolution.h"

__global__ void convolve_rows_kernel_naive(float *result_dev, float *c_dev, float *kernel_dev, int num_rows, int num_cols, int half_width)
{
	int threadX = threadIdx.x;
	int threadY = threadIdx.y;
	int blockX = blockIdx.x;
	int blockY = blockIdx.y;

	int row = blockY*blockDim.y + threadY;
	int col = blockX*blockDim.x + threadX;

	int k = 0;
	int i;

	if (col >= num_cols || row >= num_rows)
		return;

	float sum = 0.0f;
	sum = 0.0f;
	for (i = col-half_width; i <= col+half_width; i++) {
		if (i < 0 || i >= num_cols)
			continue;
		sum += c_dev[row*num_cols + i]*kernel_dev[i+half_width-col];
	}

	result_dev[row*num_cols+col] = sum;
	k += blockDim.x;

    return;
}

__global__ void convolve_columns_kernel_naive(float *result_dev, float *c_dev, float *kernel_dev, int num_rows, int num_cols, int half_width)
{
	int threadX = threadIdx.x;
	int threadY = threadIdx.y;
	int blockX = blockIdx.x;
	int blockY = blockIdx.y;

	int row = blockY*blockDim.y + threadY;
	int col = blockX*blockDim.x + threadX;

	int k = 0;
	int i;

	if (col >= num_cols || row >= num_rows)
		return;

	float sum = 0.0f;
	sum = 0.0f;
	for (i = row-half_width; i <= row+half_width; i++) {
		if (i < 0 || i >= num_rows)
			continue;
		sum += c_dev[i*num_cols + col]*kernel_dev[i+half_width-row];
	}
	result_dev[row*num_cols+col] = sum;
	k += blockDim.y;

    return;
}

__global__ void convolve_rows_kernel_opt(float *result_dev, float *c_dev, float *kernel_dev, int num_rows, int num_cols, int half_width)
{
	int threadX = threadIdx.x;
	int threadY = threadIdx.y;
	int blockX = blockIdx.x;
	int blockY = blockIdx.y;

	int row = blockY*blockDim.y + threadY;
	int col = blockX*blockDim.x + threadX;

	__shared__ float preload_c[TILE_SIZE][TILE_SIZE+2*HALF_WIDTH];

	int k = 0;
	int i;
	float sum = 0.0f;
	// load right halo into preload_c arr
	if (threadX < half_width) {
		if (col + blockDim.x >= num_cols)
			preload_c[threadY][half_width + blockDim.x + threadX] = 0.0f;
		else
			preload_c[threadY][half_width + blockDim.x + threadX] = c_dev[row*num_cols + col + blockDim.x];
	}
	// load left halo into preload_c arr
	if (threadX > blockDim.x - half_width) {
		if ((int)(col - blockDim.x) < 0)
			preload_c[threadY][half_width - blockDim.x + threadX] = 0.0f;
		else
			preload_c[threadY][half_width - blockDim.x + threadX] = c_dev[row*num_cols + col - blockDim.x];
	}
	// load center
	if (col >= num_cols || row >= num_rows) {
		preload_c[threadY][threadX+half_width] = 0.0f;
		return;
	} else {
		preload_c[threadY][threadX+half_width] = c_dev[row*num_cols+col];
	}


	sum = 0.0f;
	for (i = 0; i < 2*half_width + 1; i++) {
		sum += preload_c[threadY][threadX + i]*const_kernel_dev[i];
	}

	result_dev[row*num_cols+col] = sum;
	k += blockDim.x;

    return;
}

__global__ void convolve_columns_kernel_opt(float *result_dev, float *c_dev, float *kernel_dev, int num_rows, int num_cols, int half_width)
{
	int threadX = threadIdx.x;
	int threadY = threadIdx.y;
	int blockX = blockIdx.x;
	int blockY = blockIdx.y;

	int row = blockY*blockDim.y + threadY;
	int col = blockX*blockDim.x + threadX;

	__shared__ float preload_c[TILE_SIZE+2*HALF_WIDTH][TILE_SIZE];

	int k = 0;
	int i;
	float sum = 0.0f;
	// load bottom halo into preload_c arr
	if (threadY < half_width) {
		if (row + blockDim.y >= num_rows)
			preload_c[half_width + blockDim.y + threadY][threadX] = 0.0f;
		else
			preload_c[half_width + blockDim.y + threadY][threadX] = c_dev[(row+blockDim.y)*num_cols + col];
	}
	// load left halo into preload_c arr
	if (threadY > blockDim.y - half_width) {
		if ((int)(row - blockDim.y) < 0)
			preload_c[half_width - blockDim.y + threadY][threadX] = 0.0f;
		else
			preload_c[half_width - blockDim.y + threadY][threadX] = c_dev[(row-blockDim.y)*num_cols + col];
	}
	// load center
	if (col >= num_cols || row >= num_rows) {
		preload_c[threadY+half_width][threadX] = 0.0f;
		return;
	} else {
		preload_c[threadY+half_width][threadX] = c_dev[row*num_cols+col];
	}


	sum = 0.0f;
	for (i = 0; i < 2*half_width + 1; i++) {
		sum += preload_c[threadY + i][threadX]*const_kernel_dev[i];
	}

	result_dev[row*num_cols+col] = sum;
	k += blockDim.y;

    return;
}




