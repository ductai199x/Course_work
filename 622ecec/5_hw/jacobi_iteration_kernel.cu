#include "jacobi_iteration.h"

/* FIXME: Write the device kernels to solve the Jacobi iterations */


__global__ void jacobi_iteration_kernel_naive(float *A, float *B, float *X, float *X_new, float *partial_ssd_vec, int size)
{
	/* Obtain thread index within the thread block */
	int threadX = threadIdx.x;

	/* Obtain block index within the grid */
	int blockX = blockIdx.x;

	/* Find position in matrix */
	int column = blockDim.x * blockX + threadX;

	if (column >= size) {
		return;
	}

	int i;
	float sum = 0.0;
	float tmp_ssd = 0.0;
	for (i = 0; i < size; i++) {
		if (i != column)
			sum += A[column*size+i] * X[i];
	}

	X_new[column] = (B[column] - sum)/A[column*size+column];
	tmp_ssd = X_new[column] - X[column];
	partial_ssd_vec[column] = tmp_ssd*tmp_ssd;

    return;
}

__global__ void jacobi_iteration_kernel_optimized(float *A, float *B, float *X, float *X_new, float *partial_ssd_vec, int size)
{
	// Allocate shared memory
	__shared__ float Asub[TILE_SIZE][TILE_SIZE];
	__shared__ float Adiag[MATRIX_SIZE];
	__shared__ float Xsub[TILE_SIZE];

	int threadX = threadIdx.x;
	int threadY = threadIdx.y;

	int blockY = blockIdx.y;

	int row = blockDim.y * blockY + threadY;

	int k = 0;
	int i;
	float sum = 0.0f;
	float tmp_ssd = 0.0f;

	if (row >= size) {
		return;
	}

	while (k < size) {
		// load elements of A into Asub
		Asub[threadY][threadX] = A[row * size + k + threadX];

		// load elements of A into Adiag
		if (row == k + threadX)
			Adiag[row] = Asub[threadY][threadX];

		// load elements of X into Xsub
		if(threadY == 0)
			Xsub[threadX] = X[k + threadX];

		__syncthreads();

		if (threadX == 0) {
			for (i = 0; i < TILE_SIZE; i++) {
				if (row != k+i) {
					sum += Asub[threadY][i] * Xsub[i];
				}
			}
		}

		__syncthreads();

		k += TILE_SIZE;
	}

	if (threadX == 0) {
		X_new[row] = (B[row] - sum)/Adiag[row];
		tmp_ssd = X_new[row] - X[row];
		partial_ssd_vec[row] = tmp_ssd * tmp_ssd;
	}

    return;
}
