/*
 * counting_sort.h
 *
 *  Created on: May 29, 2020
 *      Author: sweet
 */

#ifndef COUNTING_SORT_H_
#define COUNTING_SORT_H_

#include <inttypes.h>

#define MIN_VALUE 0
#define MAX_VALUE 255

#define THREAD_BLK_SIZE 256

void print_histogram(int *bin, int num_bins, int num_elements);

#endif /* COUNTING_SORT_H_ */
