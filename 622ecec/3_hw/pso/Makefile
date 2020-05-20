# Simple makefile for building PSO on xunil
# Author: Naga Kandasamy, May 5, 2020

CC		:= /usr/bin/gcc
CCFLAGS := -fopenmp -std=c99 -Wall -O3 
LDLIBS := -lm -lpthread

all: clean pso

pso: pso.o pso_utils.o optimize_gold.o optimize_using_omp.o 
	$(CC) -o pso.out pso.o pso_utils.o optimize_gold.o optimize_using_omp.o $(LDLIBS) $(CCFLAGS)

pso.o: pso.c pso.h
	$(CC) -c pso.c $(CCFLAGS)

pso_utils.o: pso_utils.c pso.h
	$(CC) -c pso_utils.c $(CCFLAGS)

optimize_gold.o: optimize_gold.c pso.h
	$(CC) -c optimize_gold.c $(CCFLAGS)

optimize_using_omp.o: optimize_using_omp.c pso.h 
	$(CC) -c optimize_using_omp.c $(CCFLAGS)

holder_table: pso
	./pso.out holder_table 2 64 -10 10 10000 8

eggholder: pso
	./pso.out eggholder 2 64 -512 512 10000 8

booth: pso
	./pso.out booth 2 64 -10 10 10000 8

clean: 
	rm -rf pso.out *.o > /dev/null 2>&1


