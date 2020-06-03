/*
 * seperable_convolution.h
 *
 *  Created on: May 29, 2020
 *      Author: sweet
 */

#ifndef SEPERABLE_CONVOLUTION_H_
#define SEPERABLE_CONVOLUTION_H_

#define HALF_WIDTH 8
#define COEFF 2
#define TILE_SIZE 32

__constant__ float const_kernel_dev[HALF_WIDTH*2+1];

#endif /* SEPERABLE_CONVOLUTION_H_ */
