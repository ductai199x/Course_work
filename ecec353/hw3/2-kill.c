/* This simple example demonstrates how to send a signal from one
 * process to another.
 *
 * Specifically, this example:
 *   -- creates a child process
 *   -- installs a SIGINT signal handler for the child
 *   -- sends SIGINT to the child from the parent using kill()
 *   -- reaps the child using waitpid()
 *
 *   Author: James A. Shackleford
 *     Date: February 18th, 2016
 */

#include <signal.h>    /* kill(), signal(), SIGINT      */
#include <sys/wait.h>  /* waitpid()                     */
#include <unistd.h>    /* fork(), sleep()               */
#include <stdlib.h>    /* exit(), sleep(), EXIT_SUCCESS */
#include <stdio.h>     /* print(), fflush(), stdout     */

void child_handler(int sig)
{
    if (sig == SIGINT) {
        printf("\nChild: Caught SIGINT! (exiting)\n\n");
        exit(EXIT_SUCCESS);
    }
}

int main(int argc, char** argv)
{
    pid_t pid;
    void (*old_handler)(int sig);
    int ret, i;

    pid = fork();
    if (pid == 0) {
        old_handler = signal(SIGINT, child_handler);
        while(1);
    }

    printf("Parent: Sending SIGINT to child in\n");
    for (i=5; i>0; i--) {
        printf ("%i... ", i);
        fflush(stdout);
        sleep(1);
    }

    kill(pid, SIGINT);

    printf ("\nParent: Waiting on Child\n");
    waitpid(pid, NULL, 0);
    printf ("Parent: Done Waiting. (exiting)\n");

    return 0;
}
