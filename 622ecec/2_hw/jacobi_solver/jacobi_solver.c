/* Code for the Jacobi method of solving a system of linear equations 
 * by iteration.

 * Author: Naga Kandasamy
 * Date modified: April 22, 2020
 *
 * Compile as follows:
 * gcc -o jacobi_solver jacobi_solver.c compute_gold.c -Wall -O3 -lpthread -lm 
 * 
 * Student names: Tai D. Nguyen
 * Date: May 4, 2020
*/

#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <semaphore.h>
#include <pthread.h>
#include <sys/time.h>
#include <getopt.h>
#include "jacobi_solver.h"

/* Uncomment the line below to spit out debug information */ 
// #define DEBUG

#define TOLERANCE 0.00001

typedef struct thread_data_s {
    int tid;                        /* The thread ID */
    int num_threads;                /* Number of threads in the worker pool */
	matrix_t *A;
	matrix_t *B;
	matrix_t *X;
	matrix_t *new_X;
	double *partial_ssd;
} thread_data_t;

typedef struct barrier_s {
    pthread_mutex_t mutex;          /* Protects access to the counter */
    pthread_cond_t wait;          /* Signals that barrier is safe to cross */
    int counter;                /* The value itself */
} barrier_t;

int num_threads = 1;
barrier_t barrier;  

void barrier_sync(barrier_t *, int, int);
void *pthread_jacobi_solver(void *args);

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
    num_threads = t;

    if (rpt_mode)
        printf("%u\t%d\t", matrix_size, num_threads);

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
	compute_using_pthreads(&A, &mt_solution_x, &B);
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

/* FIXME: Complete this function to perform the Jacobi calculation using pthreads. 
 * Result must be placed in mt_sol_x. */

void compute_using_pthreads (matrix_t *A, matrix_t *X, matrix_t *B)
{
	// create new_mt_sol_x
	matrix_t *new_X = (matrix_t*)malloc(sizeof(matrix_t));
	new_X->num_columns = X->num_columns;
	new_X->num_rows = X->num_rows;

	int size_X = X->num_columns*X->num_rows;
	new_X->elements = (float*)malloc(sizeof(float)*size_X);

	memcpy(new_X->elements, X->elements, size_X*sizeof(float));


	// initialize barrier sync
	barrier.counter = 0;
	pthread_mutex_init(&(barrier.mutex), NULL);
	pthread_cond_init(&(barrier.wait), NULL);


	// create threads
	pthread_t *thread_id = (pthread_t *)malloc (num_threads * sizeof(pthread_t));

    pthread_attr_t attributes;      /* Thread attributes */
    pthread_attr_init(&attributes); /* Initialize thread attributes to default values */
	

	// malloc for partial ssds
	double *partial_ssd = (double*)malloc(sizeof(double)*num_threads);

    /* Fork point: allocate memory on heap for required data structures and create worker threads */
    thread_data_t *thread_data = (thread_data_t *) malloc(sizeof(thread_data_t) * num_threads);	  

	int i;
    for (i = 0; i < num_threads; i++) {
        thread_data[i].tid = i; 
        thread_data[i].num_threads = num_threads;
        thread_data[i].A = A; 
		thread_data[i].B = B; 
        thread_data[i].X = X; 
		thread_data[i].new_X = new_X;
		thread_data[i].partial_ssd = partial_ssd;
    }

	for (i = 0; i < num_threads; i++)
        pthread_create(&thread_id[i], &attributes, pthread_jacobi_solver, (void *)&thread_data[i]);

	for (i = 0; i < num_threads; i++)
        pthread_join(thread_id[i], NULL);

	free(thread_id);
    free(thread_data);

}

int done = 0;

void *pthread_jacobi_solver(void *args)
{
	thread_data_t *thread_data = (thread_data_t *)args;

	float sum = 0;
	int sum_chunk_size = thread_data->A->num_columns;
	
	int stride_X = thread_data->num_threads;

	while(!done) {

		float tmp_ssd = 0;
		double partial_ssd = 0;
		for (int i = thread_data->tid; i < thread_data->A->num_rows; i+=stride_X) {
			sum = 0;
			for (int j = 0; j < sum_chunk_size; j++) {
				if (i != j) {
					sum += thread_data->A->elements[i*sum_chunk_size + j] * thread_data->X->elements[j];
				}
			}
			thread_data->new_X->elements[i] = (thread_data->B->elements[i] - sum)/thread_data->A->elements[i*sum_chunk_size + i];
			
			tmp_ssd = thread_data->X->elements[i] - thread_data->new_X->elements[i];

			partial_ssd += tmp_ssd*tmp_ssd;
		}

		thread_data->partial_ssd[thread_data->tid] = partial_ssd;

		// barrier sync
		barrier_sync(&barrier, thread_data->tid, thread_data->num_threads);

		// do the sum ssds
		if (thread_data->tid == 0) {
			double sum_ssds = 0;
			for (int i = 0; i < thread_data->num_threads; i++) {
				sum_ssds += thread_data->partial_ssd[i];
			}
			if (sqrt(sum_ssds) < 0.00001) {
				done = 1;
			}
		}

		barrier_sync(&barrier, thread_data->tid, thread_data->num_threads);

		if (!done) {
			matrix_t *tmp_ptr;
			tmp_ptr = thread_data->X;
			thread_data->X = thread_data->new_X;
			thread_data->new_X = tmp_ptr;
		}
	}

	pthread_exit(NULL);

}

/* Barrier synchronization implementation */
void barrier_sync(barrier_t *barrier, int tid, int num_threads)
{
	pthread_mutex_lock(&(barrier->mutex));
	if (barrier->counter == num_threads - 1) {		

		barrier->counter = 0;
		pthread_cond_broadcast(&(barrier->wait));
	} 
	else {
		barrier->counter++;
		pthread_cond_wait(&(barrier->wait), &(barrier->mutex));
	}

	pthread_mutex_unlock(&(barrier->mutex));
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
			fprintf(stderr, "%f ", M.elements[i * M.num_rows + j]);
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



