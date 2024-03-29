#ifndef _VECTOR_REDUCTION_KERNEL_H_
#define _VECTOR_REDUCTION_KERNEL_H_

#define THREAD_BLOCK_SIZE 1024          /* Size of a thread block */
#define NUM_BLOCKS 40                   /* Number of thread blocks */

/* Use compare and swap to acquire mutex */
__device__ void lock(int *mutex) 
{	  
    while (atomicCAS(mutex, 0, 1) != 0);
    return;
}

/* Use atomic exchange operation to release mutex */
__device__ void unlock(int *mutex) 
{
    atomicExch(mutex, 0);
    return;
}

__global__ void vector_reduction_kernel(float *A, double *result, 
                                        int num_elements, int *mutex)
{
	__shared__ double sum_per_thread[THREAD_BLOCK_SIZE];    /* Shared memory for thread block */
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;  /* Obtain thread index */
	int stride = blockDim.x * gridDim.x;                    /* Stride for each thread. */
	double sum = 0.0f; 
	unsigned int i = thread_id; 

	/* Generate the partial sum */
	while (i < num_elements) {
		sum += (double) A[i];
		i += stride;
	}

	sum_per_thread[threadIdx.x] = sum;      /* Copy sum to shared memory */
	__syncthreads ();                       /* Wait for all threads in thread block to finish */

	/* Reduce values generated by thread block to a single value. We assume that 
       the number of threads per block is power of two. */
	i = blockDim.x/2;
	while (i != 0) {
		if (threadIdx.x < i) 
			sum_per_thread[threadIdx.x] += sum_per_thread[threadIdx.x + i];
		__syncthreads();
		i /= 2;
	}

	/* Accumulate sum computed by this thread block into the global shared variable */
	if (threadIdx.x == 0) {
		lock(mutex);
		*result += sum_per_thread[0];
		unlock(mutex);
	}
}

#endif /* _VECTOR_REDUCTION_KERNEL_H_ */
