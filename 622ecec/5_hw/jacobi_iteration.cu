/* Host code for the Jacobi method of solving a system of linear equations 
 * by iteration.

 * Build as follws: make clean && make

 * Author: Naga Kandasamy
 * Date modified: May 21, 2020
*/

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "jacobi_iteration.h"

/* Include the kernel code */
#include "jacobi_iteration_kernel.cu"

/* Uncomment the line below if you want the code to spit out debug information. */ 
/* #define DEBUG */


#define MAX_ITER 10000

int main(int argc, char **argv) 
{
	if (argc > 1) {
		printf("This program accepts no arguments\n");
		exit(EXIT_FAILURE);
	}

    matrix_t  A;                    /* N x N constant matrix */
	matrix_t  B;                    /* N x 1 b matrix */
	matrix_t reference_x;           /* Reference solution */ 
	matrix_t gpu_naive_solution_x;  /* Solution computed by naive kernel */
    matrix_t gpu_opt_solution_x;    /* Solution computed by optimized kernel */

	/* Initialize the random number generator */
	srand(time(NULL));

	/* Generate diagonally dominant matrix */ 
    printf("\nGenerating %d x %d system\n", MATRIX_SIZE, MATRIX_SIZE);
	A = create_diagonally_dominant_matrix(MATRIX_SIZE, MATRIX_SIZE);
	if (A.elements == NULL) {
        printf("Error creating matrix\n");
        exit(EXIT_FAILURE);
	}
	
    /* Create the other vectors */
    B = allocate_matrix_on_host(MATRIX_SIZE, 1, 1);
	reference_x = allocate_matrix_on_host(MATRIX_SIZE, 1, 0);
	gpu_naive_solution_x = allocate_matrix_on_host(MATRIX_SIZE, 1, 0);
    gpu_opt_solution_x = allocate_matrix_on_host(MATRIX_SIZE, 1, 0);

#ifdef DEBUG
	print_matrix(A);
	print_matrix(B);
	print_matrix(reference_x);
#endif

    /* Compute Jacobi solution on CPU */
	printf("\nPerforming Jacobi iteration on the CPU\n");
	struct timeval start, stop;
	gettimeofday(&start, NULL);
    compute_gold(A, reference_x, B);
	gettimeofday(&stop, NULL);
	fprintf(stderr, "Execution time CPU GOLD = %fs\n", (float) (stop.tv_sec - start.tv_sec
				+ (stop.tv_usec - start.tv_usec) / (float) 1000000));
//    display_jacobi_solution(A, reference_x, B); /* Display statistics */
	
	/* Compute Jacobi solution on device. Solutions are returned 
       in gpu_naive_solution_x and gpu_opt_solution_x. */
    printf("\nPerforming Jacobi iteration on device\n");
    compute_on_device_naive(A, gpu_naive_solution_x, B);
	compute_on_device_opt(A, gpu_opt_solution_x, B);
//    display_jacobi_solution(A, gpu_naive_solution_x, B); /* Display statistics */
//    display_jacobi_solution(A, gpu_opt_solution_x, B);
    
    /* Check if pthread result matches reference solution within specified tolerance */
    fprintf(stderr, "\nChecking results gpu naive\n");
    int size = reference_x.num_rows;
    int res = check_results(reference_x.elements, gpu_naive_solution_x.elements, size, THRESHOLD);
    fprintf(stderr, "TEST %s\n", (0 == res) ? "PASSED" : "FAILED");

    fprintf(stderr, "\nChecking results gpu optimized\n");
    res = check_results(reference_x.elements, gpu_opt_solution_x.elements, size, THRESHOLD);
    fprintf(stderr, "TEST %s\n", (0 == res) ? "PASSED" : "FAILED");


    free(A.elements); 
	free(B.elements); 
	free(reference_x.elements); 
	free(gpu_naive_solution_x.elements);
    free(gpu_opt_solution_x.elements);
	
    exit(EXIT_SUCCESS);
}


/* FIXME: Complete this function to perform Jacobi calculation on device */
void compute_on_device_naive(const matrix_t A, matrix_t sol, const matrix_t B)
{
	// allocate x_new and partial_ssd_vec on host
	matrix_t X_new = allocate_matrix_on_host(MATRIX_SIZE, 1, 0);
	matrix_t partial_ssd_vec = allocate_matrix_on_host(MATRIX_SIZE, 1, 0);

    // allocate resources on device, then copy them to device
	matrix_t A_dev = allocate_matrix_on_device(A);
	matrix_t B_dev = allocate_matrix_on_device(B);
	matrix_t X_dev = allocate_matrix_on_device(sol);
	matrix_t X_new_dev = allocate_matrix_on_device(X_new);
	matrix_t PSV_dev = allocate_matrix_on_device(partial_ssd_vec);

	copy_matrix_to_device(A_dev, A);
	copy_matrix_to_device(B_dev, B);
	copy_matrix_to_device(X_dev, sol);
	copy_matrix_to_device(X_new_dev, X_new);
	copy_matrix_to_device(PSV_dev, partial_ssd_vec);

    // set up the execution grid on device
	dim3 thread_block(THREAD_BLOCK_SIZE, 1);
	dim3 grid((MATRIX_SIZE + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE, 1);
	fprintf(stderr, "Setting up a %d x 1 grid of thread blocks\n",
			(MATRIX_SIZE + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE);

	int iter = 0;
	float sum_ssds = 0.0;
	int done = 0;

	struct timeval start, stop;
	gettimeofday(&start, NULL);

	while(!done && iter < MAX_ITER) {
		// launch kernel
		if (iter % 2 == 0)
			jacobi_iteration_kernel_naive<<<grid, thread_block>>>(A_dev.elements, B_dev.elements, X_dev.elements,
					X_new_dev.elements, PSV_dev.elements, MATRIX_SIZE);
		else
			jacobi_iteration_kernel_naive<<<grid, thread_block>>>(A_dev.elements, B_dev.elements, X_new_dev.elements,
					X_dev.elements, PSV_dev.elements, MATRIX_SIZE);
		cudaDeviceSynchronize();

		// check error on device
		check_CUDA_error("Error in kernel");

		// copy result from device to host
		copy_matrix_from_device(sol, X_dev);
		copy_matrix_from_device(X_new, X_new_dev);
		copy_matrix_from_device(partial_ssd_vec, PSV_dev);

		sum_ssds = 0.0;
		for (int i = 0; i < MATRIX_SIZE; i++) {
			sum_ssds += partial_ssd_vec.elements[i];
		}

		if (sqrt(sum_ssds) < THRESHOLD) { done = 1; }
		iter++;
	}

//	printf("iter: %d\n", iter);

	gettimeofday(&stop, NULL);
	fprintf(stderr, "Execution time CUDA NAIVE = %fs\n", (float) (stop.tv_sec - start.tv_sec
				+ (stop.tv_usec - start.tv_usec) / (float) 1000000));
//	printf("%.5f\n", (float) (stop.tv_sec - start.tv_sec
//						+ (stop.tv_usec - start.tv_usec) / (float) 1000000));

    // free up resources
	free_matrix_on_device(&X_dev);
	free_matrix_on_device(&X_new_dev);
	free_matrix_on_device(&PSV_dev);

	free_matrix_on_host(&X_new);
	free_matrix_on_host(&partial_ssd_vec);

    return;
}


void compute_on_device_opt(const matrix_t A, matrix_t sol, const matrix_t B)
{
	// allocate x_new and partial_ssd_vec on host
	matrix_t X_new = allocate_matrix_on_host(MATRIX_SIZE, 1, 0);
	matrix_t partial_ssd_vec = allocate_matrix_on_host(MATRIX_SIZE, 1, 0);

    // allocate resources on device, then copy them to device
	matrix_t A_dev = allocate_matrix_on_device(A);
	matrix_t B_dev = allocate_matrix_on_device(B);
	matrix_t X_dev = allocate_matrix_on_device(sol);
	matrix_t X_new_dev = allocate_matrix_on_device(X_new);
	matrix_t PSV_dev = allocate_matrix_on_device(partial_ssd_vec);

	copy_matrix_to_device(A_dev, A);
	copy_matrix_to_device(B_dev, B);
	copy_matrix_to_device(X_dev, sol);
	copy_matrix_to_device(X_new_dev, X_new);
	copy_matrix_to_device(PSV_dev, partial_ssd_vec);

    // set up the execution grid on device
	dim3 thread_block(TILE_SIZE, TILE_SIZE);
//	fprintf(stderr, "Setting up a %d x %d grid of thread blocks\n",
//			(MATRIX_SIZE + TILE_SIZE - 1) / TILE_SIZE,
//			(MATRIX_SIZE + TILE_SIZE - 1) / TILE_SIZE);
	dim3 grid(1, (MATRIX_SIZE + TILE_SIZE - 1) / TILE_SIZE);


	int iter = 0;
	float sum_ssds = 0.0;
	int done = 0;

	struct timeval start, stop;
	gettimeofday(&start, NULL);

	while(!done && iter < MAX_ITER) {
		// launch kernel
		if (iter % 2 == 0)
			jacobi_iteration_kernel_optimized<<<grid, thread_block>>>(A_dev.elements, B_dev.elements, X_dev.elements,
					X_new_dev.elements, PSV_dev.elements, MATRIX_SIZE);
		else
			jacobi_iteration_kernel_optimized<<<grid, thread_block>>>(A_dev.elements, B_dev.elements, X_new_dev.elements,
					X_dev.elements, PSV_dev.elements, MATRIX_SIZE);
		cudaDeviceSynchronize();

		// check error on device
		check_CUDA_error("Error in kernel");

		// copy result from device to host
		copy_matrix_from_device(sol, X_dev);
		copy_matrix_from_device(X_new, X_new_dev);
		copy_matrix_from_device(partial_ssd_vec, PSV_dev);

		sum_ssds = 0.0;
		for (int i = 0; i < MATRIX_SIZE; i++) {
			sum_ssds += partial_ssd_vec.elements[i];
		}

		if (sqrt(sum_ssds) < THRESHOLD) { done = 1; }
		iter++;
	}

//	printf("iter: %d\n", iter);

	gettimeofday(&stop, NULL);
	fprintf(stderr, "Execution time CUDA OPT = %fs\n", (float) (stop.tv_sec - start.tv_sec
				+ (stop.tv_usec - start.tv_usec) / (float) 1000000));
//	printf("%.5f\n", (float) (stop.tv_sec - start.tv_sec
//						+ (stop.tv_usec - start.tv_usec) / (float) 1000000));

    // free up resources
	free_matrix_on_device(&X_dev);
	free_matrix_on_device(&X_new_dev);
	free_matrix_on_device(&PSV_dev);

	free_matrix_on_host(&X_new);
	free_matrix_on_host(&partial_ssd_vec);

    return;
}

/* Allocate matrix on the device of same size as M */
matrix_t allocate_matrix_on_device(const matrix_t M)
{
    matrix_t Mdevice = M;
    int size = M.num_rows * M.num_columns * sizeof(float);
    cudaMalloc((void **)&Mdevice.elements, size);
    return Mdevice;
}

/* Allocate a matrix of dimensions height * width.
   If init == 0, initialize to all zeroes.  
   If init == 1, perform random initialization.
*/
matrix_t allocate_matrix_on_host(int num_rows, int num_columns, int init)
{	
    matrix_t M;
    M.num_columns = num_columns;
    M.num_rows = num_rows;
    int size = M.num_rows * M.num_columns;
		
	M.elements = (float *)malloc(size * sizeof(float));
	for (unsigned int i = 0; i < size; i++) {
		if (init == 0) 
            M.elements[i] = 0; 
		else
            M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
	}
    
    return M;
}

/* Free matrix on device */
void free_matrix_on_device(matrix_t *M) {
	cudaFree(M->elements);
	M->elements = NULL;
}

/* Free matrix on host */
void free_matrix_on_host(matrix_t *M) {
	free(M->elements);
	M->elements = NULL;
}


/* Copy matrix to device */
void copy_matrix_to_device(matrix_t Mdevice, const matrix_t Mhost)
{
    int size = Mhost.num_rows * Mhost.num_columns * sizeof(float);
    Mdevice.num_rows = Mhost.num_rows;
    Mdevice.num_columns = Mhost.num_columns;
    cudaMemcpy(Mdevice.elements, Mhost.elements, size, cudaMemcpyHostToDevice);
    return;
}

/* Copy matrix from device to host */
void copy_matrix_from_device(matrix_t Mhost, const matrix_t Mdevice)
{
    int size = Mdevice.num_rows * Mdevice.num_columns * sizeof(float);
    cudaMemcpy(Mhost.elements, Mdevice.elements, size, cudaMemcpyDeviceToHost);
    return;
}

/* Prints the matrix out to screen */
void print_matrix(const matrix_t M)
{
	for (unsigned int i = 0; i < M.num_rows; i++) {
        for (unsigned int j = 0; j < M.num_columns; j++) {
			printf("%f ", M.elements[i * M.num_columns + j]);
        }
		
        printf("\n");
	} 
	
    printf("\n");
    return;
}

/* Returns a floating-point value between [min, max] */
float get_random_number(int min, int max)
{
    float r = rand()/(float)RAND_MAX;
	return (float)floor((double)(min + (max - min + 1) * r));
}

/* Check for errors in kernel execution */
void check_CUDA_error(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if ( cudaSuccess != err) {
		printf("CUDA ERROR: %s (%s).\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}	
    
    return;    
}

/* Create diagonally dominant matrix */
matrix_t create_diagonally_dominant_matrix(unsigned int num_rows, unsigned int num_columns)
{
	matrix_t M;
	M.num_columns = num_columns;
	M.num_rows = num_rows; 
	unsigned int size = M.num_rows * M.num_columns;
	M.elements = (float *)malloc(size * sizeof(float));
    if (M.elements == NULL)
        return M;

	/* Create a matrix with random numbers between [-.5 and .5] */
    unsigned int i, j;
	for (i = 0; i < size; i++)
        M.elements[i] = get_random_number (MIN_NUMBER, MAX_NUMBER);
	
	/* Make diagonal entries large with respect to the entries on each row. */
	for (i = 0; i < num_rows; i++) {
		float row_sum = 0.0;		
		for (j = 0; j < num_columns; j++) {
			row_sum += fabs(M.elements[i * M.num_rows + j]);
		}
		
        M.elements[i * M.num_rows + i] = 0.5 + row_sum;
	}

    return M;
}

/* Check if results generated by cpu and gpu match within tolerance */
int check_results(float *A, float *B, int size, float tolerance)
{
    int i;
    for (i = 0; i < size; i++)
        if(fabsf(A[i] - B[i]) > tolerance)
            return -1;
    return 0;
}
