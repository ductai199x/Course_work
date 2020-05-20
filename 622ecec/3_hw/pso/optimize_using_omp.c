/* Implementation of PSO using OpenMP.
 *
 * Author: Naga Kandasamy
 * Date: May 2, 2020
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include "pso.h"

int pso_solve_omp(char *function, swarm_t *swarm, float xmax, float xmin, int num_iter, int num_threads);

int optimize_using_omp(char *function, swarm_t *swarm, float xmin, float xmax, int num_iter, int num_threads)
{
    /* Solve PSO */
    int g; 
    struct timeval start, stop;	
	gettimeofday(&start, NULL);
    g = pso_solve_omp(function, swarm, xmax, xmin, num_iter, num_threads);
    gettimeofday(&stop, NULL);
	float exec_time = (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec)/(float)1000000);
    if (g >= 0) {
        fprintf(stderr, "Solution OMP:\n");
        pso_print_particle(&swarm->particle[g]);
        fprintf(stderr, "Execution time = %fs\n", exec_time);
    }

    return g;
}

int pso_solve_omp(char *function, swarm_t *swarm, float xmax, float xmin, int num_iter, int num_threads)
{
    int i, j, iter, g;
    float w, c1, c2;
    float r1, r2;
    float curr_fitness;
    particle_t *particle, *gbest;

    float best_fitness = INFINITY;

    w = 0.79;
    c1 = 1.49;
    c2 = 1.49;
    iter = 0;
    g = -1;

    unsigned seed; 

    while (iter < num_iter) {
        #pragma omp parallel num_threads(num_threads) shared(swarm) 
        {
            int tid = omp_get_thread_num();
            seed = 25234 + 17*tid;
            #pragma omp for private(i, j, particle, gbest, curr_fitness, seed)
            for (i = 0; i < swarm->num_particles; i++) {
                curr_fitness = 0;
                particle = &swarm->particle[i];
                gbest = &swarm->particle[particle->g];  /* Best performing particle from last iteration */ 

                for (j = 0; j < particle->dim; j++) {   /* Update this particle's state */
                    r1 = (float)rand_r(&seed)/(float)RAND_MAX;
                    r2 = (float)rand_r(&seed)/(float)RAND_MAX;
                    /* Update particle velocity */
                    particle->v[j] = w * particle->v[j]\
                                    + c1 * r1 * (particle->pbest[j] - particle->x[j])\
                                    + c2 * r2 * (gbest->x[j] - particle->x[j]);
                    /* Clamp velocity */
                    if ((particle->v[j] < -fabsf(xmax - xmin)) || (particle->v[j] > fabsf(xmax - xmin))) 
                        particle->v[j] = uniform(-fabsf(xmax - xmin), fabsf(xmax - xmin));

                    /* Update particle position */
                    particle->x[j] = particle->x[j] + particle->v[j];
                    if (particle->x[j] > xmax)
                        particle->x[j] = xmax;
                    if (particle->x[j] < xmin)
                        particle->x[j] = xmin;

                    // printf("%d %d %d\n", tid, i, j); fflush(stdout);
                } /* State update */
                
                /* Evaluate current fitness */
                
                pso_eval_fitness(function, particle, &curr_fitness);

                /* Update pbest */
                if (curr_fitness < particle->fitness) {
                    particle->fitness = curr_fitness;
                    memmove(particle->pbest, particle->x, sizeof(float)*particle->dim);
                }
            }

            #pragma omp barrier
            
            #pragma omp for reduction(min:best_fitness)
            for(i = 0 ; i < swarm->num_particles; i++) {
                if ((&swarm->particle[i])->fitness < best_fitness) {
                    best_fitness = (&swarm->particle[i])->fitness;
                }
            }

            #pragma omp barrier

            #pragma omp for private(i)
            for(i = 0 ; i < swarm->num_particles; i++) {
                if ((&swarm->particle[i])->fitness == best_fitness)
                    g = i;  
            }

            #pragma omp barrier

            #pragma omp for private(i)
            for (i = 0; i < swarm->num_particles; i++) {
                (&swarm->particle[i])->g = g;
            }
        }

        

        iter++;
    }
    return g;
}

