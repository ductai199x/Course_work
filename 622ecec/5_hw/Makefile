# A simple CUDA makefile
#
# Author: Naga Kandasamy
# Date: 3/9/2018
#
# CUDA depends on two things:
#  1) The CUDA nvcc compiler, which needs to be on your path,
#	or called directly, which we do here
#  2) The CUDA shared library being available at runtime,
#	which we make available by setting the LD_LIBRARY_PATH
#	variable for the duration of the makefile.
#   
#   Note that you can set your PATH and LD_LIBRARY_PATH variables as part of your
# .bash_profile so that you can compile and run without using this makefile.

SOURCE		:= jacobi_iteration.cu jacobi_iteration_gold.cpp 
HEADER		:= $(wildcard *.h)
NVCCFLAGS	:= -O3 -gencode arch=compute_75,code=sm_75
NVCC		:= /usr/local/cuda/bin/nvcc
LD_LIBRARY_PATH	:= /usr/local/cuda/lib64
TARGET		:= jacobi_iteration.o

all: clean $(TARGET)

$(TARGET): $(SOURCE) $(HEADER)
	$(NVCC) -o $(TARGET) $(SOURCE) -I./ $(NVCCFLAGS)

clean:
	rm -rf $(TARGET)