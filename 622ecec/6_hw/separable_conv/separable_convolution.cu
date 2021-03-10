/* Host code that implements a  separable convolution filter of a 
 * 2D signal with a gaussian kernel.
 * 
 * Author: Naga Kandasamy
 * Date modified: May 26, 2020
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "seperable_convolution.h"

extern "C" void compute_gold(float *, float *, int, int, int);
extern "C" float *create_kernel(float, int);
void print_kernel(float *, int);
void print_matrix(float *, int, int);

/* Uncomment line below to spit out debug information */
//#define DEBUG

/* Include device code */
#include "separable_convolution_kernel.cu"

void compute_on_device(float *gpu_result, float *matrix_c,\
                   float *kernel, int num_cols,\
                   int num_rows, int half_width, int is_opt);

int main(int argc, char **argv)
{
    if (argc < 3) {
        printf("Usage: %s num-rows num-columns\n", argv[0]);
        printf("num-rows: height of the matrix\n");
        printf("num-columns: width of the matrix\n");
        exit(EXIT_FAILURE);
    }

    int num_rows = atoi(argv[1]);
    int num_cols = atoi(argv[2]);

    /* Create input matrix */
    int num_elements = num_rows * num_cols;
    printf("Creating input matrix of %d x %d\n", num_rows, num_cols);
    float *matrix_a = (float *)malloc(sizeof(float) * num_elements);
    float *matrix_c = (float *)malloc(sizeof(float) * num_elements);
	
    srand(time(NULL));
    int i;
    for (i = 0; i < num_elements; i++) {
        matrix_a[i] = rand()/(float)RAND_MAX;			 
        matrix_c[i] = matrix_a[i]; /* Copy contents of matrix_a into matrix_c */
    }
	 
	/* Create Gaussian kernel */	  
    float *gaussian_kernel = create_kernel((float)COEFF, HALF_WIDTH);	
//    print_kernel(gaussian_kernel, HALF_WIDTH);
	  
    /* Convolve matrix along rows and columns. 
       The result is stored in matrix_a, thereby overwriting the 
       original contents of matrix_a.		
     */
    printf("\nConvolving the matrix on the CPU\n");
	struct timeval start, stop;
	gettimeofday(&start, NULL);
    compute_gold(matrix_a, gaussian_kernel, num_cols,\
                  num_rows, HALF_WIDTH);
    gettimeofday(&stop, NULL);
	fprintf(stderr, "Execution time CPU GOLD = %fs\n", (float) (stop.tv_sec - start.tv_sec
				+ (stop.tv_usec - start.tv_usec) / (float) 1000000));

    float *gpu_result = (float *)malloc(sizeof(float) * num_elements);
    
    /* FIXME: Edit this function to complete the functionality on the GPU.
       The input matrix is matrix_c and the result must be stored in 
       gpu_result.
     */
    printf("\nConvolving matrix on the GPU naive\n");
    compute_on_device(gpu_result, matrix_c, gaussian_kernel, num_cols,\
                       num_rows, HALF_WIDTH, 0);
       
    printf("\nComparing CPU and GPU results\n");
    float sum_delta = 0, sum_ref = 0;
    for (i = 0; i < num_elements; i++) {
        sum_delta += fabsf(matrix_a[i] - gpu_result[i]);
        sum_ref   += fabsf(matrix_a[i]);
    }
        
    float L1norm = sum_delta / sum_ref;
    float eps = 1e-6;
    printf("L1 norm: %E\n", L1norm);
    printf((L1norm < eps) ? "TEST PASSED\n" : "TEST FAILED\n");

    // ----------------------------------------------------------------
    printf("\nConvolving matrix on the GPU optimized\n");
    compute_on_device(gpu_result, matrix_c, gaussian_kernel, num_cols,\
                       num_rows, HALF_WIDTH, 1);

    printf("\nComparing CPU and GPU results\n");
    for (i = 0; i < num_elements; i++) {
        sum_delta += fabsf(matrix_a[i] - gpu_result[i]);
        sum_ref   += fabsf(matrix_a[i]);
    }

    printf("L1 norm: %E\n", L1norm);
    printf((L1norm < eps) ? "TEST PASSED\n" : "TEST FAILED\n");

    free(matrix_a);
    free(matrix_c);
    free(gpu_result);
    free(gaussian_kernel);

    exit(EXIT_SUCCESS);
}

/* FIXME: Edit this function to compute the convolution on the device.*/
void compute_on_device(float *gpu_result, float *matrix_c,\
                   float *kernel, int num_cols,\
                   int num_rows, int half_width, int is_opt)
{
	int size = num_rows*num_cols*sizeof(float);
	int kernel_size = (half_width*2+1)*sizeof(float);

	float* kernel_dev;
	float* result_dev;
	float* c_dev;
	// allocate memory on device
	cudaMalloc((void **)&result_dev, size);
	cudaMalloc((void **)&c_dev, size);
	cudaMalloc((void **)&kernel_dev, kernel_size);

	// copy to device
	cudaMemcpy(result_dev, gpu_result, size, cudaMemcpyHostToDevice);
	cudaMemcpy(c_dev, matrix_c, size, cudaMemcpyHostToDevice);
	if (is_opt)
		cudaMemcpyToSymbol(const_kernel_dev, kernel, kernel_size);
	else
		cudaMemcpy(kernel_dev, kernel, kernel_size, cudaMemcpyHostToDevice);

    // set up the execution grid on device
	dim3 thread_block(32, 32);
	dim3 grid((num_rows + 32 - 1) / 32, (num_cols + 32 - 1) / 32);
	printf("Creating grid of %d x %d", (num_rows + 32 - 1) / 32, (num_cols + 32 - 1) / 32);

	// timer start
	struct timeval start, stop;
	gettimeofday(&start, NULL);

	// launch kernel
	if (is_opt)
		convolve_rows_kernel_opt<<<grid, thread_block>>>(result_dev, c_dev, kernel_dev, num_rows, num_cols, half_width);
	else
		convolve_rows_kernel_naive<<<grid, thread_block>>>(result_dev, c_dev, kernel_dev, num_rows, num_cols, half_width);
	cudaDeviceSynchronize();
	cudaMemcpy(gpu_result, result_dev, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(c_dev, gpu_result, size, cudaMemcpyHostToDevice);

	//switch x, y of grid
//	grid.x = 1; grid.y = (num_rows + 32 - 1) / 32;
	if (is_opt)
		convolve_columns_kernel_opt<<<grid, thread_block>>>(result_dev, c_dev, kernel_dev, num_rows, num_cols, half_width);
	else
		convolve_columns_kernel_naive<<<grid, thread_block>>>(result_dev, c_dev, kernel_dev, num_rows, num_cols, half_width);
	cudaDeviceSynchronize();

	// timer stop
    gettimeofday(&stop, NULL);
    if (is_opt)
		fprintf(stderr, "Execution time GPU OPT = %fs\n", (float) (stop.tv_sec - start.tv_sec
					+ (stop.tv_usec - start.tv_usec) / (float) 1000000));
    else
    	fprintf(stderr, "Execution time GPU NAIVE = %fs\n", (float) (stop.tv_sec - start.tv_sec
					+ (stop.tv_usec - start.tv_usec) / (float) 1000000));

	// copy result from device to host
	cudaMemcpy(gpu_result, result_dev, size, cudaMemcpyDeviceToHost);

	cudaFree(kernel_dev);
	cudaFree(result_dev);
	cudaFree(c_dev);

    return;
}


/* Check for errors reported by the CUDA run time */
void check_for_error(char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		printf("CUDA ERROR: %s (%s)\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

    return;
} 

/* Print convolution kernel */
void print_kernel(float *kernel, int half_width)
{
    int i, j = 0;
    for (i = -half_width; i <= half_width; i++) {
        printf("%0.2f ", kernel[j]);
        j++;
    }

    printf("\n");
    return;
}

/* Print matrix */
void print_matrix(float *matrix, int num_cols, int num_rows)
{
    int i,  j;
    float element;
    for (i = 0; i < num_rows; i++) {
        for (j = 0; j < num_cols; j++){
            element = matrix[i * num_cols + j];
            printf("%0.2f ", element);
        }
        printf("\n");
    }

    return;
}

