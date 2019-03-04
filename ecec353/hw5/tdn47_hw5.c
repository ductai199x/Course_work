#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>

#define NUM_WORKER_THREADS 20

struct thread_data {
    pthread_t tid;
    unsigned int num;
};

struct worker_state {
    int still_working;
    pthread_mutex_t mutex;
    pthread_cond_t signal;
};

static struct worker_state wstate = {
    .still_working = NUM_WORKER_THREADS,
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .signal = PTHREAD_COND_INITIALIZER,
};

static unsigned int result[NUM_WORKER_THREADS];

void* verify_total_thread (void* param)
{
    unsigned long long total = 0;
    int i;
    for (i = 0; i < NUM_WORKER_THREADS; i++) {
        total += result[i];
    }
    printf("Verifying Total: %llu\n", total);

    pthread_exit(0);
}

void* signal_thread (void* param)
{
    while ( wstate.still_working ) {
        pthread_cond_wait(&wstate.signal, &wstate.mutex);
    }

    unsigned long long total = 0;
    int i;
    for (i = 0; i < NUM_WORKER_THREADS; i++) {
        total += result[i];
    }
    printf("Total: \t\t %llu\n", total);

    pthread_exit(0);
}

void* worker_thread (void* param)
{
    struct thread_data* data = (struct thread_data*) param;

    pthread_mutex_lock(&wstate.mutex);
    
    result[data->num] = ((int)data->tid)*((int)data->tid);
    wstate.still_working = wstate.still_working - 1;
    
    pthread_mutex_unlock(&wstate.mutex);

    pthread_cond_broadcast(&wstate.signal);

    pthread_exit(0);
}

int main (int argc, char** argv)
{
    struct thread_data* threads;
    threads = malloc(sizeof(*threads)*NUM_WORKER_THREADS);
    pthread_t threads_arr[NUM_WORKER_THREADS];
    pthread_t sig_thread;

    int i;

    for (i = 0; i < NUM_WORKER_THREADS; i++) {
        threads[i].num = i;
        pthread_create(&threads_arr[i], NULL, worker_thread, &threads[i]);
        threads[i].tid = threads_arr[i];
    }

    pthread_create(&sig_thread, NULL, signal_thread, NULL);
    pthread_join(sig_thread, NULL);
  
    free(threads);
    
    pthread_t verify_total;
    pthread_create(&verify_total, NULL, verify_total_thread, NULL);
    pthread_join(verify_total, NULL);
 
    return 0;
}
