#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <signal.h>

#define FALSE 0
#define TRUE 1
#define NUM_PRIME 10000000
#define INTERVAL 10
#define PRINTPID 0

int num_found = 0;
int primes[NUM_PRIME];

int is_prime(unsigned int num);
void find_primes();
void print_result();
void sigalrm_handler(int sig);
void sigusr1_handler(int sig);
void sigquit_handler(int sig);
void sigterm_handler(int sig);


void sigalrm_handler(int sig)
{
    print_result();
    alarm(INTERVAL);
}

void sigusr1_handler(int sig)
{
    print_result();
}

void sigquit_handler(int sig)
{
    
}

void sigterm_handler(int sig)
{
    print_result();
    printf("Goodbye!\n");

    exit(EXIT_SUCCESS);
}

int is_prime(unsigned int num)
{
    if ( num < 2 ) return FALSE;
    if ( num == 2 ) return TRUE;
    if ( num % 2 == 0 ) return FALSE;

    int i;
    for ( i = 3; i < num / 2; i += 2 ) {
        if ( num % i == 0 )
            return FALSE;
    }
    return TRUE;
}

void find_primes()
{
    sigset_t block;
    sigset_t prev_mask;

    sigemptyset(&block);
    sigaddset(&block, SIGINT);
    sigaddset(&block, SIGALRM);
    sigaddset(&block, SIGUSR1);
    sigaddset(&block, SIGTERM);
    sigaddset(&block, SIGQUIT);

    alarm(INTERVAL);
    int i = 0;
    while ( num_found < NUM_PRIME ) {
        if ( is_prime(i) ) {
            sigprocmask(SIG_BLOCK, &block, &prev_mask);
            primes[num_found] = i;
            num_found++;
            sigprocmask(SIG_SETMASK, &prev_mask, NULL);
        }
        i++; 
    }
}

void print_result()
{
    int a1, a2, a3, a4, a5;

    // Just in case!
    a1 = num_found == 0 ? num_found : num_found-1;
    a2 = num_found < 2 ? num_found : num_found-2;
    a3 = num_found < 3 ? num_found : num_found-3;
    a4 = num_found < 4 ? num_found : num_found-4;
    a5 = num_found < 5 ? num_found : num_found-5;

    printf("Found %i primes.\nLast 5 primes found:\n%i %i %i %i %i\n", num_found, primes[a1], primes[a2], primes[a3], primes[a4], primes[a5]);
}

int main(int argc, char** argv)
{
    signal(SIGALRM, sigalrm_handler);
    signal(SIGUSR1, sigusr1_handler);
    signal(SIGQUIT, sigquit_handler);
    signal(SIGTERM, sigterm_handler);

    if ( PRINTPID )
        printf("is_prime--running at pid=%i\n", getpid());

    find_primes();

    return 0;
}
