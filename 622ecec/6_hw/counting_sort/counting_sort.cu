/* Host-side code to perform counting sort 
 * Author: Naga Kandasamy
 * Date modified: May 27, 2020
 * 
 * Compile as follows: make clean && make
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include "counting_sort.h"

#include "counting_sort_kernel.cu"

/* Uncomment to spit out debug info */
// #define DEBUG

extern "C" int counting_sort_gold(int *, int *, int, int);
int rand_int(int, int);
void print_array(int *, int);
void print_min_and_max_in_array(int *, int);
void compute_on_device(int *, int *, int, int);
int check_if_sorted(int *, int);
int compare_results(int *, int *, int);

int main(int argc, char **argv)
{
    if (argc < 2) {
        printf("Usage: %s num-elements\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    int num_elements = atoi(argv[1]);
    int range = MAX_VALUE - MIN_VALUE;
    int *input_array, *sorted_array_reference, *sorted_array_d;

    /* Populate input array with random integers between [0, RANGE] */
    printf("Generating input array with %d elements in the range 0 to %d\n", num_elements, range);
    input_array = (int *)malloc(num_elements * sizeof(int));
    if (input_array == NULL) {
        perror("malloc");
        exit(EXIT_FAILURE);
    }
    
    srand(time(NULL));
    int i;
    for (i = 0; i < num_elements; i++)
        input_array[i] = rand_int (MIN_VALUE, MAX_VALUE);

#ifdef DEBUG
    print_array(input_array, num_elements);
    print_min_and_max_in_array(input_array, num_elements);
#endif

    /* Sort elements in input array using reference implementation. 
     * The result is placed in sorted_array_reference. */
    printf("\nSorting array on CPU\n");
    int status;
    sorted_array_reference = (int *)malloc(num_elements * sizeof(int));
    if (sorted_array_reference == NULL) {
        perror("malloc"); 
        exit(EXIT_FAILURE);
    }
    memset(sorted_array_reference, 0, num_elements);
	struct timeval start, stop;
	gettimeofday(&start, NULL);
    status = counting_sort_gold(input_array, sorted_array_reference, num_elements, range);
    gettimeofday(&stop, NULL);
   	fprintf(stderr, "Execution time CPU GOLD = %fs\n", (float) (stop.tv_sec - start.tv_sec
   				+ (stop.tv_usec - start.tv_usec) / (float) 1000000));
    if (status == -1) {
        exit(EXIT_FAILURE);
    }

    status = check_if_sorted(sorted_array_reference, num_elements);
    if (status == -1) {
        printf("Error sorting the input array using the reference code\n");
        exit(EXIT_FAILURE);
    }

    printf("Counting sort was successful on the CPU\n");

#ifdef DEBUG
    print_array(sorted_array_reference, num_elements);
#endif

    /* FIXME: Write function to sort elements in the array in parallel fashion. 
     * The result should be placed in sorted_array_mt. */
    printf("\nSorting array on GPU\n");
    sorted_array_d = (int *)malloc(num_elements * sizeof(int));
    if (sorted_array_d == NULL) {
        perror("malloc");
        exit(EXIT_FAILURE);
    }
    memset(sorted_array_d, 0, num_elements);
    compute_on_device(input_array, sorted_array_d, num_elements, range);

    /* Check the two results for correctness */
    printf("\nComparing CPU and GPU results\n");
    status = compare_results(sorted_array_reference, sorted_array_d, num_elements);
    if (status == 0)
        printf("Test passed\n");
    else
        printf("Test failed\n");

    exit(EXIT_SUCCESS);
}


/* FIXME: Write the GPU implementation of counting sort */
void compute_on_device(int *input_array, int *sorted_array, int num_elements, int range)
{
	int *histogram = (int*)malloc(range*sizeof(int));

	int *input_dev;
	int *sorted_dev;
	int *global_hist_dev;

	// allocate memory on device
	cudaMalloc((void**)&input_dev, num_elements*sizeof(int));
	cudaMalloc((void**)&sorted_dev, num_elements*sizeof(int));

	cudaMalloc((void**)&global_hist_dev, range*sizeof(int));
	cudaMemset(global_hist_dev, 0, range*sizeof(int));

	// copy to device
	cudaMemcpy(input_dev, input_array, num_elements*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(sorted_dev, sorted_array, num_elements*sizeof(int), cudaMemcpyHostToDevice);

	// set up the execution grid on device
	dim3 thread_block(THREAD_BLK_SIZE,1);
	dim3 grid((num_elements+THREAD_BLK_SIZE-1)/THREAD_BLK_SIZE, 1);

	// start timer
	struct timeval start, stop;
	gettimeofday(&start, NULL);

	// launch kernel
	counting_sort_kernel<<<grid, thread_block>>>(input_dev, sorted_dev, global_hist_dev, num_elements);

	// copy hist back to host
	cudaMemcpy(histogram, global_hist_dev, range*sizeof(int), cudaMemcpyDeviceToHost);

	int i;
    for (i = 1; i < range + 1; i++)
    	histogram[i] = histogram[i - 1] + histogram[i];

    int j;
    int start_idx = 0;
    for (i = 0; i < range + 1; i++) {
        for (j = start_idx; j < histogram[i]; j++) {
            sorted_array[j] = i;
        }
        start_idx = histogram[i];
    }

    gettimeofday(&stop, NULL);
	fprintf(stderr, "Execution time GPU = %fs\n", (float) (stop.tv_sec - start.tv_sec
				+ (stop.tv_usec - start.tv_usec) / (float) 1000000));

    return;
}

/* Check if array is sorted */
int check_if_sorted(int *array, int num_elements)
{
    int status = 0;
    int i;
    for (i = 1; i < num_elements; i++) {
        if (array[i - 1] > array[i]) {
            status = -1;
            break;
        }
    }

    return status;
}

/* Check if the arrays elements are identical */ 
int compare_results(int *array_1, int *array_2, int num_elements)
{
    int status = 0;
    int i;
    for (i = 0; i < num_elements; i++) {
        if (array_1[i] != array_2[i]) {
            status = -1;
            break;
        }
    }

    return status;
}

/* Return random integer between [min, max] */ 
int rand_int(int min, int max)
{
    float r = rand()/(float)RAND_MAX;
    return (int)floorf(min + (max - min) * r);
}

/* Print given array */
void print_array(int *this_array, int num_elements)
{
    printf("Array: ");
    int i;
    for (i = 0; i < num_elements; i++)
        printf("%d ", this_array[i]);
    
    printf("\n");
    return;
}

/* Return min and max values in given array */
void print_min_and_max_in_array(int *this_array, int num_elements)
{
    int i;

    int current_min = INT_MAX;
    for (i = 0; i < num_elements; i++)
        if (this_array[i] < current_min)
            current_min = this_array[i];

    int current_max = INT_MIN;
    for (i = 0; i < num_elements; i++)
        if (this_array[i] > current_max)
            current_max = this_array[i];

    printf("Minimum value in the array = %d\n", current_min);
    printf("Maximum value in the array = %d\n", current_max);
    return;
}

/* Print histogram contents */
void print_histogram(int *bin, int num_bins, int num_elements)
{
    int num_histogram_entries = 0;
    int i;

    for (i = 0; i < num_bins; i++) {
        printf("Bin %d: %d\n", i, bin[i]);
        num_histogram_entries += bin[i];
    }

    printf("Number of elements in the input array = %d \n", num_elements);
    printf("Number of histogram elements = %d \n", num_histogram_entries);

    return;
}
