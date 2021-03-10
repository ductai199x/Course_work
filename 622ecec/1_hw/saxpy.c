/* Implementation of the SAXPY loop.
 *
 * Compile as follows: gcc -o saxpy saxpy.c -O3 -Wall -std=c99 -lpthread -lm
 *
 * Author: Naga Kandasamy
 * Date created: April 14, 2020
 * Date modified: 
 *
 * Student names: Tai D. Nguyen
 * Date: April 20, 2020
 *
 * */

#define _REENTRANT /* Make sure the library functions are MT (muti-thread) safe */
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <pthread.h>
#include <unistd.h>
#include <inttypes.h>

/* Function prototypes */
void compute_gold(float *, float *, float, int);
void compute_using_pthreads_v1(float *, float *, float, int, int);
void compute_using_pthreads_v2(float *, float *, float, int, int);
int check_results(float *, float *, int, float);

void *chunk_saxpy(void *args);
void *stride_saxpy(void *args);


// Threads' data structure
typedef struct saxpy_tdata_s {
    int tid;                        /* The thread ID */
    int num_threads;                /* Number of threads in the worker pool */
    int num_elements;               /* Number of elements in the vector */
    float a;                        /* a */
    float *x;                       /* Pointer to x */
    float *y;                       /* Pointer to y */
    int offset;                     /* Starting offset for each thread within the vectors */ 
    int chunk_size;                 /* Chunk size */
}saxpy_tdata_t;

int main(int argc, char **argv)
{

    int e = 1;
    int t = 1;
    int rpt_mode = 0;

    int opt; 
      
    // put ':' in the starting of the 
    // string so that program can  
    // distinguish between '?' and ':'  
    while((opt = getopt(argc, argv, ":e:t:r")) != -1)  
    {  
        switch(opt)  
        { 
            case 'e':  
                e = atoi(optarg);
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
                abort();
        }  
    }  
	
    uint32_t num_elements = (uint32_t)pow(10, e); 
    int num_threads = t;

    if (rpt_mode)
        printf("%u\t%d\t", num_elements, t);

    float exec_time = 0;

	/* Create vectors X and Y and fill them with random numbers between [-.5, .5] */
    if (!rpt_mode)
        fprintf(stderr, "Generating input vectors\n");
    int i;
	float *x = (float *)malloc(sizeof(float) * num_elements);
    float *y1 = (float *)malloc(sizeof(float) * num_elements);              /* For the reference version */
	float *y2 = (float *)malloc(sizeof(float) * num_elements);              /* For pthreads version 1 */
	float *y3 = (float *)malloc(sizeof(float) * num_elements);              /* For pthreads version 2 */

	srand(time(NULL)); /* Seed random number generator */
	for (i = 0; i < num_elements; i++) {
		x[i] = rand()/(float)RAND_MAX - 0.5;
		y1[i] = rand()/(float)RAND_MAX - 0.5;
        y2[i] = y1[i]; /* Make copies of y1 for y2 and y3 */
        y3[i] = y1[i]; 
	}

    float a = 2.5;  /* Choose some scalar value for a */

	/* Calculate SAXPY using the reference solution. The resulting values are placed in y1 */
    if (!rpt_mode) {
        fprintf(stderr, "\nCalculating SAXPY using reference solution\n");
    }

	struct timeval start, stop;	
	gettimeofday(&start, NULL);
	
    compute_gold(x, y1, a, num_elements); 
	
    gettimeofday(&stop, NULL);
    exec_time = (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec)/(float)1000000);
    if (!rpt_mode)
	    fprintf(stderr, "Execution time = %fs\n", exec_time);
    else
        printf("%f\t", exec_time);

	/* Compute SAXPY using pthreads, version 1. Results must be placed in y2 */
    if (!rpt_mode) {
        fprintf(stderr, "\nCalculating SAXPY using pthreads, version 1\n");
    }
	gettimeofday(&start, NULL);

	compute_using_pthreads_v1(x, y2, a, num_elements, num_threads);
	
    gettimeofday(&stop, NULL);
	exec_time = (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec)/(float)1000000);
    if (!rpt_mode)
	    fprintf(stderr, "Execution time = %fs\n", exec_time);
    else
        printf("%f\t", exec_time);

    /* Compute SAXPY using pthreads, version 2. Results must be placed in y3 */
    if (!rpt_mode) {
        fprintf(stderr, "\nCalculating SAXPY using pthreads, version 2\n");
    }
	gettimeofday(&start, NULL);

	compute_using_pthreads_v2(x, y3, a, num_elements, num_threads);
	
    gettimeofday(&stop, NULL);
	exec_time = (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec)/(float)1000000);
    if (!rpt_mode)
	    fprintf(stderr, "Execution time = %fs\n", exec_time);
    else
        printf("%f\n", exec_time);

    /* Check results for correctness */
    if (!rpt_mode) {
        fprintf(stderr, "\nChecking results for correctness\n");
    }
    float eps = 1e-12;  /* Do not change this value */
    if (check_results(y1, y2, num_elements, eps) == 0) {
        if (!rpt_mode)
            fprintf(stderr, "TEST PASSED\n");
    } else {
        fprintf(stderr, "TEST FAILED\n");
    }
 
    if (check_results(y1, y3, num_elements, eps) == 0) {
        if (!rpt_mode)
            fprintf(stderr, "TEST PASSED\n");
    } else {
        fprintf(stderr, "TEST FAILED\n");
    }

	/* Free memory */ 
	free((void *)x);
	free((void *)y1);
    free((void *)y2);
	free((void *)y3);

    exit(EXIT_SUCCESS);
}

/* Compute reference soution using a single thread */
void compute_gold(float *x, float *y, float a, int num_elements)
{
	int i;
    for (i = 0; i < num_elements; i++)
        y[i] = a * x[i] + y[i]; 
}

/* Calculate SAXPY using pthreads, version 1. Place result in the Y vector */
void compute_using_pthreads_v1(float *x, float *y, float a, int num_elements, int num_threads)
{
    // allocate memory for threads
    pthread_t *thread_arr = (pthread_t *)malloc(num_threads*sizeof(pthread_t));

    // allocate memory for threads' data structures
    saxpy_tdata_t *thread_data_arr = (saxpy_tdata_t *)malloc(num_threads*sizeof(saxpy_tdata_t));

    // initialize threads' attributes
    pthread_attr_t attributes;
    pthread_attr_init(&attributes);

    // compute chunk size for each threads
    int chunk_size = num_elements/num_threads;

    for (int i = 0; i < num_threads; i++) {
        thread_data_arr[i].tid = i; 
        thread_data_arr[i].num_threads = num_threads;
        thread_data_arr[i].num_elements = num_elements; 
        thread_data_arr[i].a = a;
        thread_data_arr[i].x = x; 
        thread_data_arr[i].y = y; 
        thread_data_arr[i].offset = i * chunk_size; 
        thread_data_arr[i].chunk_size = chunk_size;
    }

    // create threads
    for (int i = 0; i < num_threads; i++) {
        pthread_create(&thread_arr[i], &attributes, chunk_saxpy, (void *)&thread_data_arr[i]);
    }

    // wait for all threads to finish
    for (int i = 0; i < num_threads; i++) {
        pthread_join(thread_arr[i], NULL);
    }

    free(thread_data_arr);
    free(thread_arr);
}

void *chunk_saxpy(void *args) {
    saxpy_tdata_t *tdata = (saxpy_tdata_t *)args;
    float a = tdata->a;
    int start = tdata->offset;
    int finish = tdata->offset + tdata->chunk_size;

    for(int i = start; i < finish; i++) {
        tdata->y[i] = a*tdata->x[i] + tdata->y[i];
    }

    pthread_exit(NULL);
}

/* Calculate SAXPY using pthreads, version 2. Place result in the Y vector */
void compute_using_pthreads_v2(float *x, float *y, float a, int num_elements, int num_threads)
{
    // allocate memory for threads
    pthread_t *thread_arr = (pthread_t *)malloc(num_threads*sizeof(pthread_t));

    // allocate memory for threads' data structures
    saxpy_tdata_t *thread_data_arr = (saxpy_tdata_t *)malloc(num_threads*sizeof(saxpy_tdata_t));

    // initialize threads' attributes
    pthread_attr_t attributes;
    pthread_attr_init(&attributes);

    // compute chunk size for each threads
    int chunk_size = num_elements/num_threads;

    for (int i = 0; i < num_threads; i++) {
        thread_data_arr[i].tid = i; 
        thread_data_arr[i].num_threads = num_threads;
        thread_data_arr[i].num_elements = num_elements; 
        thread_data_arr[i].a = a;
        thread_data_arr[i].x = x; 
        thread_data_arr[i].y = y; 
        thread_data_arr[i].offset = i * chunk_size; 
        thread_data_arr[i].chunk_size = chunk_size;
    }

    // create threads
    for (int i = 0; i < num_threads; i++) {
        pthread_create(&thread_arr[i], &attributes, stride_saxpy, (void *)&thread_data_arr[i]);
    }

    // wait for all threads to finish
    for (int i = 0; i < num_threads; i++) {
        pthread_join(thread_arr[i], NULL);
    }

    free(thread_data_arr);
    free(thread_arr);
}

void *stride_saxpy(void *args) {
    saxpy_tdata_t *tdata = (saxpy_tdata_t *)args;
    float a = tdata->a;
    int start = tdata->tid;
    int finish = tdata->num_elements;
    int stride = tdata->num_threads;

    for(int i = start; i < finish; i+=stride) {
        tdata->y[i] = a*tdata->x[i] + tdata->y[i];
    }

    pthread_exit(NULL);
}

/* Perform element-by-element check of vector if relative error is within specified threshold */
int check_results(float *A, float *B, int num_elements, float threshold)
{
    int i;
    for (i = 0; i < num_elements; i++) {
        if (fabsf((A[i] - B[i])/A[i]) > threshold)
            return -1;
    }
    
    return 0;
}



