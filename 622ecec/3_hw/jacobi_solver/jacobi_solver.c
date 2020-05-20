/* Code for the Jacobi method of solving a system of linear equations 
 * by iteration.

 * Author: Naga Kandasamy
 * Date modified: April 29, 2020
 *
 * Compile as follows:
 * gcc -o jacobi_solver jacobi_solver.c compute_gold.c -fopenmp -std=c99 -Wall -O3 -lm 
*/

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <getopt.h>
#include "jacobi_solver.h"

/* Uncomment the line below to spit out debug information */ 
/* #define DEBUG */

int thread_count = 4;
int max_iter = 100000;

int main(int argc, char **argv) 
{
	int s = 4;
    int t = 1;
    int rpt_mode = 0;

    int opt; 
      
    // put ':' in the starting of the 
    // string so that program can  
    // distinguish between '?' and ':'  
    while((opt = getopt(argc, argv, ":s:t:r")) != -1)  
    {  
        switch(opt)  
        { 
            case 's':  
                s = atoi(optarg);
                break;  
            case 't':  
                t = atoi(optarg);
                break;
            case 'r':
                rpt_mode = 1;
                break;
            case ':':  
                printf("option needs a value\n");  
                break;  
            case '?':  
                printf("unknown option: %c\n", optopt); 
                break;  
            default:
				fprintf(stderr, "Usage: %s -s matrix_size -t num_threads -r (enable reporting mode)\n", argv[0]);
                abort();
        }  
    }  
	
    int matrix_size = s; 
    thread_count = t;

    if (rpt_mode)
        printf("%u\t%d\t", matrix_size, thread_count);

    float exec_time = 0;

    matrix_t  A;                    /* N x N constant matrix */
	matrix_t  B;                    /* N x 1 b matrix */
	matrix_t reference_x;           /* Reference solution */ 
    matrix_t mt_solution_x;         /* Solution computed by pthread code */

	/* Generate diagonally dominant matrix */
	if (!rpt_mode) {
    	fprintf(stderr, "\nCreating input matrices\n");
	}
	srand(time(NULL));
	A = create_diagonally_dominant_matrix(matrix_size, matrix_size);
	if (A.elements == NULL) {
        fprintf(stderr, "Error creating matrix\n");
        exit(EXIT_FAILURE);
	}
	
    /* Create other matrices */
    B = allocate_matrix(matrix_size, 1, 1);
	reference_x = allocate_matrix(matrix_size, 1, 0);
	mt_solution_x = allocate_matrix(matrix_size, 1, 0);

#ifdef DEBUG
	print_matrix(A);
	print_matrix(B);
	print_matrix(reference_x);
#endif

    /* Compute Jacobi solution using reference code */
	if (!rpt_mode) {
		fprintf(stderr, "Generating solution using reference code\n");
	}
    int max_iter = 100000; /* Maximum number of iterations to run */
	struct timeval start, stop;	
	gettimeofday(&start, NULL);
    compute_gold(A, reference_x, B, max_iter);
	gettimeofday(&stop, NULL);
	exec_time = (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec)/(float)1000000);
    if (!rpt_mode) {
	    fprintf(stderr, "Execution time = %fs\n", exec_time);
		display_jacobi_solution(A, reference_x, B); /* Display statistics */
	} else {
        printf("%f\t", exec_time);
	}
    
	
	/* Compute the Jacobi solution using pthreads. 
     * Solutions are returned in mt_solution_x.
     * */
	if (!rpt_mode) {
    	fprintf(stderr, "\nPerforming Jacobi iteration using pthreads\n");
	}
	gettimeofday(&start, NULL);
	compute_using_omp(&A, &mt_solution_x, &B);
	gettimeofday(&stop, NULL);
	exec_time = (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec)/(float)1000000);
    if (!rpt_mode) {
	    fprintf(stderr, "Execution time = %fs\n", exec_time);
		display_jacobi_solution(A, reference_x, B); /* Display statistics */
	} else {
        printf("%f\n", exec_time);
	}
    
    free(A.elements); 
	free(B.elements); 
	free(reference_x.elements); 
	free(mt_solution_x.elements);
	
    exit(EXIT_SUCCESS);
}

/* FIXME: Complete this function to perform the Jacobi calculation using openMP. 
 * Result must be placed in mt_sol_x. */
void compute_using_omp(matrix_t *A, matrix_t *X, matrix_t *B)
{

	// create new_mt_sol_x
	matrix_t *new_X = (matrix_t*)malloc(sizeof(matrix_t));
	new_X->num_columns = X->num_columns;
	new_X->num_rows = X->num_rows;

	int size_X = X->num_columns*X->num_rows;
	new_X->elements = (float*)malloc(sizeof(float)*size_X);

	memcpy(new_X->elements, X->elements, size_X*sizeof(float));

	// malloc for partial ssds
	double *partial_ssd_vect = (double*)malloc(sizeof(double)*thread_count);

	int iter = 0; int done = 0;
	
	while(!done && iter < max_iter) {
		iter++;
		int i,j;
		double sum, partial_ssd;
		float tmp_ssd;
		#pragma omp parallel num_threads(thread_count) private(i, j, sum, partial_ssd, tmp_ssd) shared(A, X, new_X, B, partial_ssd_vect)
		{
			int tid = omp_get_thread_num();
			sum = 0; partial_ssd = 0; tmp_ssd = 0;
			for (i = tid; i < A->num_rows; i+=thread_count) {
				sum = 0; partial_ssd = 0; tmp_ssd = 0;
				for (j = 0; j < A->num_columns; j++) {
					if (i != j) {
						sum += A->elements[i*A->num_columns + j] * X->elements[j];
					}
				}
				new_X->elements[i] = (B->elements[i] - sum)/A->elements[i*A->num_columns + i];
				
				tmp_ssd = X->elements[i] - new_X->elements[i];

				partial_ssd += tmp_ssd*tmp_ssd;
			}

			partial_ssd_vect[tid] = partial_ssd;
		}

		double sum_ssds = 0;
		for (int i = 0; i < thread_count; i++) {
			sum_ssds += partial_ssd_vect[i];
		}
		if (sqrt(sum_ssds) < 0.000001) {
			done = 1;
		}

		if (!done) {
			matrix_t *tmp_ptr;
			tmp_ptr = X;
			X = new_X;
			new_X = tmp_ptr;
		}

	}

}

/* Allocate a matrix of dimensions height * width.
   If init == 0, initialize to all zeroes.  
   If init == 1, perform random initialization.
*/
matrix_t allocate_matrix(int num_rows, int num_columns, int init)
{
    int i;    
    matrix_t M;
    M.num_columns = num_columns;
    M.num_rows = num_rows;
    int size = M.num_rows * M.num_columns;
		
	M.elements = (float *)malloc(size * sizeof(float));
	for (i = 0; i < size; i++) {
		if (init == 0) 
            M.elements[i] = 0; 
		else
            M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
	}
    
    return M;
}	

/* Print matrix to screen */
void print_matrix(const matrix_t M)
{
    int i, j;
	for (i = 0; i < M.num_rows; i++) {
        for (j = 0; j < M.num_columns; j++) {
			fprintf(stderr, "%f ", M.elements[i * M.num_columns + j]);
        }
		
        fprintf(stderr, "\n");
	} 
	
    fprintf(stderr, "\n");
    return;
}

/* Return a floating-point value between [min, max] */
float get_random_number(int min, int max)
{
    float r = rand ()/(float)RAND_MAX;
	return (float)floor((double)(min + (max - min + 1) * r));
}

/* Check if matrix is diagonally dominant */
int check_if_diagonal_dominant(const matrix_t M)
{
    int i, j;
	float diag_element;
	float sum;
	for (i = 0; i < M.num_rows; i++) {
		sum = 0.0; 
		diag_element = M.elements[i * M.num_rows + i];
		for (j = 0; j < M.num_columns; j++) {
			if (i != j)
				sum += abs(M.elements[i * M.num_rows + j]);
		}
		
        if (diag_element <= sum)
			return -1;
	}

	return 0;
}

/* Create diagonally dominant matrix */
matrix_t create_diagonally_dominant_matrix (int num_rows, int num_columns)
{
	matrix_t M;
	M.num_columns = num_columns;
	M.num_rows = num_rows; 
	int size = M.num_rows * M.num_columns;
	M.elements = (float *)malloc(size * sizeof(float));

    int i, j;
	fprintf(stderr, "Generating %d x %d matrix with numbers between [-.5, .5]\n", num_rows, num_columns);
	for (i = 0; i < size; i++)
        M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
	
	/* Make diagonal entries large with respect to the entries on each row. */
    float row_sum;
	for (i = 0; i < num_rows; i++) {
		row_sum = 0.0;		
		for (j = 0; j < num_columns; j++) {
			row_sum += fabs(M.elements[i * M.num_rows + j]);
		}
		
        M.elements[i * M.num_rows + i] = 0.5 + row_sum;
	}

    /* Check if matrix is diagonal dominant */
	if (check_if_diagonal_dominant(M) < 0) {
		free(M.elements);
		M.elements = NULL;
	}
	
    return M;
}



