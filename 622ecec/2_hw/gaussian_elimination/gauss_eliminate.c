/* Gaussian elimination code.
 * 
 * Author: Naga Kandasamy
 * Date of last update: April 22, 2020
 *
 * Student names(s): FIXME
 * Date: FIXME
 *
 * Compile as follows: 
 * gcc -o gauss_eliminate gauss_eliminate.c compute_gold.c -O3 -Wall -lpthread -lm
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
#include "gauss_eliminate.h"

#define MIN_NUMBER 2
#define MAX_NUMBER 50


typedef struct thread_data_s {
    int tid;                        /* The thread ID */
    int num_threads;                /* Number of threads in the worker pool */
	Matrix *U;
} thread_data_t;

typedef struct barrier_s {
    pthread_mutex_t mutex;          /* Protects access to the counter */
    pthread_cond_t wait;          /* Signals that barrier is safe to cross */
    int counter;                /* The value itself */
} barrier_t;

int num_threads = 1;
barrier_t barrier;  

/* Function prototypes */
extern int compute_gold(float *, int);
Matrix allocate_matrix(int, int, int);
void gauss_eliminate_using_pthreads(Matrix *);
int perform_simple_check(const Matrix);
void print_matrix(const Matrix);
float get_random_number(int, int);
int check_results(float *, float *, int, float);

void barrier_sync(barrier_t *, int, int);
void *pthread_gauss(void *args);


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

    Matrix A;			                                            /* Input matrix */
    Matrix U_reference;		                                        /* Upper triangular matrix computed by reference code */
    Matrix U_mt;			                                        /* Upper triangular matrix computed by pthreads */

    fprintf(stderr, "Generating input matrices\n");
    srand(time (NULL));                                             /* Seed random number generator */
    A = allocate_matrix(matrix_size, matrix_size, 1);               /* Allocate and populate random square matrix */
    U_reference = allocate_matrix (matrix_size, matrix_size, 0);    /* Allocate space for reference result */
    U_mt = allocate_matrix (matrix_size, matrix_size, 0);           /* Allocate space for multi-threaded result */

    /* Copy contents A matrix into U matrices */
    int i, j;
    for (i = 0; i < A.num_rows; i++) {
        for (j = 0; j < A.num_rows; j++) {
            U_reference.elements[A.num_rows * i + j] = A.elements[A.num_rows * i + j];
            U_mt.elements[A.num_rows * i + j] = A.elements[A.num_rows * i + j];
        }
    }

    fprintf(stderr, "\nPerforming gaussian elimination using reference code\n");
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    int status = compute_gold(U_reference.elements, A.num_rows);
	gettimeofday(&stop, NULL);
	exec_time = (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec)/(float)1000000);
    fprintf(stderr, "Execution time = %fs\n", exec_time);
    printf("%f\t", exec_time);

    if (status < 0) {
        fprintf(stderr, "Failed to convert given matrix to upper triangular. Try again.\n");
        exit(EXIT_FAILURE);
    }
  
    status = perform_simple_check(U_reference);	/* Check that principal diagonal elements are 1 */ 
    if (status < 0) {
        fprintf(stderr, "Upper triangular matrix is incorrect. Exiting.\n");
        exit(EXIT_FAILURE);
    }
    fprintf(stderr, "Single-threaded Gaussian elimination was successful.\n");
  
    /* FIXME: Perform Gaussian elimination using pthreads. 
     * The resulting upper triangular matrix should be returned in U_mt */
    fprintf(stderr, "\nPerforming gaussian elimination using pthreads\n");

    gettimeofday(&start, NULL);
    gauss_eliminate_using_pthreads(&U_mt);
    gettimeofday(&stop, NULL);
	exec_time = (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec)/(float)1000000);

    fprintf(stderr, "Execution time = %fs\n", exec_time);
    printf("%f\n", exec_time);

    /* Check if pthread result matches reference solution within specified tolerance */
    fprintf(stderr, "\nChecking results\n");
    int size = matrix_size * matrix_size;
    int res = check_results(U_reference.elements, U_mt.elements, size, 1e-6);
    fprintf(stderr, "TEST %s\n", (0 == res) ? "PASSED" : "FAILED");

    /* Free memory allocated for matrices */
    free(A.elements);
    free(U_reference.elements);
    free(U_mt.elements);

    exit(EXIT_SUCCESS);
}


/* FIXME: Write code to perform gaussian elimination using pthreads */
void gauss_eliminate_using_pthreads(Matrix *U)
{
    // initialize barrier sync
	barrier.counter = 0;
	pthread_mutex_init(&(barrier.mutex), NULL);
	pthread_cond_init(&(barrier.wait), NULL);


	// create threads
	pthread_t *thread_id = (pthread_t *)malloc (num_threads * sizeof(pthread_t));

    pthread_attr_t attributes;      /* Thread attributes */
    pthread_attr_init(&attributes); /* Initialize thread attributes to default values */

    /* Fork point: allocate memory on heap for required data structures and create worker threads */
    thread_data_t *thread_data = (thread_data_t *) malloc(sizeof(thread_data_t) * num_threads);	  

	int i;
    for (i = 0; i < num_threads; i++) {
        thread_data[i].tid = i; 
        thread_data[i].num_threads = num_threads;
        thread_data[i].U = U; 
    }

	for (i = 0; i < num_threads; i++)
        pthread_create(&thread_id[i], &attributes, pthread_gauss, (void *)&thread_data[i]);

	for (i = 0; i < num_threads; i++)
        pthread_join(thread_id[i], NULL);

	free(thread_id);
    free(thread_data);

}

void *pthread_gauss(void *args)
{
    thread_data_t *thread_data = (thread_data_t *)args;

    int i, j, k;
    unsigned int n_rows = thread_data->U->num_rows;
    unsigned int n_cols = thread_data->U->num_columns;
    float* mat = thread_data->U->elements;

    int tid = thread_data->tid;
    int stride = thread_data->num_threads;
    
    
    for (k = 0; k < n_rows; k++) {
        int chunk_size = (int)floor(((float)n_cols-k-1)/(float)thread_data->num_threads); 
        int offset = tid*chunk_size + k + 1;
        int start = offset;
        int finish = (tid < (thread_data->num_threads - 1)) ? offset+chunk_size : n_cols;
        for (j = start; j < finish; j++) {
            mat[n_cols * k + j] = (float)(mat[n_cols * k + j] / mat[n_cols * k + k]);
        }

        barrier_sync(&barrier, thread_data->tid, thread_data->num_threads);
        
        for (i = (k + tid + 1); i < n_rows; i+=stride) {
            for (j = (k + 1); j < n_cols; j++) {
                mat[n_cols * i + j] -= (mat[n_cols * i + k] * mat[n_cols * k + j]);	
            }
            mat[n_cols * i + k] = 0;
        }

        mat[n_cols * k + k] = 1;

        barrier_sync(&barrier, thread_data->tid, thread_data->num_threads);
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


/* Check if results generated by single threaded and multi threaded versions match within tolerance */
int check_results(float *A, float *B, int size, float tolerance)
{
    int i;
    for (i = 0; i < size; i++)
        if(fabsf(A[i] - B[i]) > tolerance)
            return -1;
    return 0;
}


/* Allocate a matrix of dimensions height*width
 * If init == 0, initialize to all zeroes.  
 * If init == 1, perform random initialization. 
*/
Matrix allocate_matrix(int num_rows, int num_columns, int init)
{
    int i;
    Matrix M;
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

/* Return a random floating-point number between [min, max] */ 
float get_random_number(int min, int max)
{
    return (float)floor((double)(min + (max - min + 1) * ((float)rand() / (float)RAND_MAX)));
}

/* Perform simple check on upper triangular matrix if the principal diagonal elements are 1 */
int perform_simple_check(const Matrix M)
{
    int i;
    for (i = 0; i < M.num_rows; i++)
        if ((fabs(M.elements[M.num_rows * i + i] - 1.0)) > 1e-6)
            return -1;
  
    return 0;
}

void print_matrix(const Matrix M)
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