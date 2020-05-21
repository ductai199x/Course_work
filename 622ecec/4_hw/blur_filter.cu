/* Reference code implementing the box blur filter.

 Build and execute as follows:
 make clean && make
 ./blur_filter size

 Author: Naga Kandasamy
 Date created: May 3, 2019
 Date modified: May 12, 2020

 FIXME: Student name(s)
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

/* #define DEBUG */

/* Include the kernel code */
#include "blur_filter_kernel.cu"

extern "C" void compute_gold(const image_t, image_t);
void compute_on_device(const image_t, image_t);
image_t allocate_matrix_on_device(image_t M);
void copy_matrix_to_device(image_t Mdevice, image_t Mhost);
void copy_matrix_from_device(image_t Mhost, image_t Mdevice);
void free_matrix_on_device(image_t *M);
void free_matrix_on_host(image_t *M);
void check_CUDA_error(const char *msg);
int check_results(const float *, const float *, int, float);
void print_image(const image_t);

int main(int argc, char **argv) {
	if (argc < 2) {
		fprintf(stderr, "Usage: %s size\n", argv[0]);
		fprintf(
		stderr,
				"size: Height of the image. The program assumes size x size image.\n");
		exit(EXIT_FAILURE);
	}

	/* Allocate memory for the input and output images */
	int size = atoi(argv[1]);

	printf("%d\t", size);

	fprintf(stderr, "Creating %d x %d images\n", size, size);
	image_t in, out_gold, out_gpu;
	in.size = out_gold.size = out_gpu.size = size;
	in.elements = (float *) malloc(sizeof(float) * size * size);
	out_gold.elements = (float *) malloc(sizeof(float) * size * size);
	out_gpu.elements = (float *) malloc(sizeof(float) * size * size);
	if ((in.elements == NULL) || (out_gold.elements == NULL)
			|| (out_gpu.elements == NULL)) {
		perror("Malloc");
		exit(EXIT_FAILURE);
	}

	/* Poplulate our image with random values between [-0.5 +0.5] */
	srand(time(NULL));
	int i;
	for (i = 0; i < size * size; i++)
		in.elements[i] = rand() / (float) RAND_MAX - 0.5;

	/* Calculate the blur on the CPU. The result is stored in out_gold. */
	fprintf(stderr, "Calculating blur on the CPU\n");
	struct timeval start, stop;
	gettimeofday(&start, NULL);
	compute_gold(in, out_gold);
	gettimeofday(&stop, NULL);
//	fprintf(stderr, "Execution time = %fs\n",
//			(float) (stop.tv_sec - start.tv_sec
//					+ (stop.tv_usec - start.tv_usec) / (float) 1000000));
	printf("%.5f\t", (float) (stop.tv_sec - start.tv_sec
					+ (stop.tv_usec - start.tv_usec) / (float) 1000000));

#ifdef DEBUG
	print_image(in);
	print_image(out_gold);
#endif

	/* FIXME: Calculate the blur on the GPU. The result is stored in out_gpu. */
	fprintf(stderr, "Calculating blur on the GPU\n");
	compute_on_device(in, out_gpu);
	//  print_image(out_gpu);

	/* Check CPU and GPU results for correctness */
	fprintf(stderr, "Checking CPU and GPU results\n");
	int num_elements = out_gold.size * out_gold.size;
	float eps = 1e-6; /* Do not change */
	int check;
	check = check_results(out_gold.elements, out_gpu.elements, num_elements,
			eps);
	if (check == 0)
		fprintf(stderr, "TEST PASSED\n");
	else
		fprintf(stderr, "TEST FAILED\n");

	/* Free data structures on the host */
	free((void *) in.elements);
	free((void *) out_gold.elements);
	free((void *) out_gpu.elements);

	exit(EXIT_SUCCESS);
}

/* FIXME: Complete this function to calculate the blur on the GPU */
void compute_on_device(const image_t in_host, image_t out_host) {
	/* Allocate memory and copy matrices to device */
	image_t in_dev = allocate_matrix_on_device(in_host);
	image_t out_dev = allocate_matrix_on_device(out_host);

	copy_matrix_to_device(in_dev, in_host);
	copy_matrix_to_device(out_dev, out_host);

	struct timeval start, stop;
	gettimeofday(&start, NULL);

	/* Set up the execution grid */
	dim3 threads(TILE_SIZE, TILE_SIZE);
	fprintf(stderr, "Setting up a %d x %d grid of thread blocks\n",
			(out_dev.size + TILE_SIZE - 1) / TILE_SIZE,
			(out_dev.size + TILE_SIZE - 1) / TILE_SIZE);
	dim3 grid((out_dev.size + TILE_SIZE - 1) / TILE_SIZE,
			(out_dev.size + TILE_SIZE - 1) / TILE_SIZE);

	/* Launch kernel */
	blur_filter_kernel<<<grid, threads>>>(in_dev.elements, out_dev.elements,
			in_host.size);
	cudaDeviceSynchronize();

	gettimeofday(&stop, NULL);
//	fprintf(stderr, "Execution time = %fs\n", (float) (stop.tv_sec - start.tv_sec
//				+ (stop.tv_usec - start.tv_usec) / (float) 1000000));
	printf("%.5f\n", (float) (stop.tv_sec - start.tv_sec
						+ (stop.tv_usec - start.tv_usec) / (float) 1000000));

	check_CUDA_error("Error in kernel");

	copy_matrix_from_device(out_host, out_dev);

	free_matrix_on_device(&in_dev);
	free_matrix_on_device(&out_dev);
}

/* Allocate memory on device for matrix */
image_t allocate_matrix_on_device(image_t M) {
	image_t Mdevice = M;
	int size = M.size * M.size * sizeof(float);

	cudaMalloc((void **) &Mdevice.elements, size);
	if (Mdevice.elements == NULL) {
		fprintf(stderr, "CudaMalloc error\n");
		exit(EXIT_FAILURE);
	}

	return Mdevice;
}

/* Copy matrix from host memory to device memory */
void copy_matrix_to_device(image_t Mdevice, image_t Mhost) {
	int size = Mhost.size * Mhost.size * sizeof(float);
	cudaMemcpy(Mdevice.elements, Mhost.elements, size, cudaMemcpyHostToDevice);
}

/* Copy matrix from device memory to host memory */
void copy_matrix_from_device(image_t Mhost, image_t Mdevice) {
	int size = Mdevice.size * Mdevice.size * sizeof(float);
	cudaMemcpy(Mhost.elements, Mdevice.elements, size, cudaMemcpyDeviceToHost);
}

/* Free matrix on device */
void free_matrix_on_device(image_t *M) {
	cudaFree(M->elements);
	M->elements = NULL;
}

/* Free matrix on host */
void free_matrix_on_host(image_t *M) {
	free(M->elements);
	M->elements = NULL;
}

/* Check for errors during kernel execution */
void check_CUDA_error(const char *msg) {
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "CUDA ERROR: %s (%s).\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

/* Check correctness of results */
int check_results(const float *pix1, const float *pix2, int num_elements,
		float eps) {
	int i;
	for (i = 0; i < num_elements; i++)
		if (fabsf((pix1[i] - pix2[i]) / pix1[i]) > eps)
			return -1;

	return 0;
}

/* Print out the image contents */
void print_image(const image_t img) {
	int i, j;
	float val;
	for (i = 0; i < img.size; i++) {
		for (j = 0; j < img.size; j++) {
			val = img.elements[i * img.size + j];
			printf("%0.4f ", val);
		}
		printf("\n");
	}

	printf("\n");
}
