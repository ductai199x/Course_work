#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>

#include "kill.h"
#include "helper.h"

void print_usage()
{
    printf("Usage: kill [-s <signal>] <pid> | %%<job> ...\n");
}

killsig_t* parse_kill_args(char** argv)
{
    killsig_t* k = malloc(sizeof(k));
    k->pids = malloc(1*sizeof(int));
    k->jobs = malloc(1*sizeof(int));
    int *pids = (int*)malloc(100 * sizeof(int));
    int *jobs = (int*)malloc(100 * sizeof(int));
    int *p, *j;
    p = pids;
    j = jobs;

    int argc = 0;
    while ( argv[argc] ) {
        argc++;
    }
    
    if ( argc < 2 ) {
        print_usage();
        return NULL;
    }
    
    int signal = SIGINT;
    if ( !strcmp(argv[1], "-s") ) {
        if ( argc < 4 ) {
            printf("pssh: invalid number of arguments.\n");
            print_usage();
            return NULL;
        }

        if ( check_int(argv[2]) ) {
            signal = atoi(argv[2]);
        }
        else {
            if ( (signal = str_to_sig(argv[2])) == -1 ) {
                printf("pssh: invalid signal: <signal>\n");
                return NULL;
            }
        }
        
        int i;
        char *start, *end, *arg;
        for ( i = 3; i < argc; i++ ) {
            if ( argv[i][0] == '%' ) {
                start = ++argv[i];
                end = (strlen(argv[i]) + argv[i]);
                arg = strndup(start, end - start);
                arg[end - start] = '\0';
                if ( check_int(arg) ) {
                    *j = atoi(arg);
                    j++;
                } else {
                    printf("pssh: invalid job number: [job number]\n");
                    return NULL;
                }
            }
            else {
                if ( check_int(argv[i]) ) {
                    *p = atoi(argv[i]);
                    p++;
                } else {
                    printf("pssh: invalid pid: [pid number]\n");
                    return NULL;
                }
            }
        }
    }
    else {
        int i;
        char *start, *end, *arg;
        for ( i = 1; i < argc; i++ ) {
            if ( argv[i][0] == '%' ) {
                start = ++argv[i];
                end = (strlen(argv[i]) + argv[i]);
                arg = strndup(start, end - start);
                arg[end - start] = '\0';
                if ( check_int(arg) ) {
                    *j = atoi(arg);
                    j++;
                } else {
                    printf("pssh: invalid job number: [job number]\n");
                    return NULL;
                }
            }
            else {
                if ( check_int(argv[i]) ) {
                    *p = atoi(argv[i]);
                    p++;
                } else {
                    printf("pssh: invalid pid: [pid number]");
                    return NULL;
                }
            }
        }
    }

    *p = -1;
    *j = -1;
    k->signal = signal;
    k->pids = pids;
    k->jobs = jobs;

    return k;
}

int send_signal(int signal, int pid)
{
    if ( kill(pid, signal) == -1 ) {
        if ( errno == EPERM ) {
            fprintf(stderr, "pssh: PID %i exists, but not able to receive signals\n", pid);
        }
        else if ( errno == EINVAL ) {
            fprintf(stderr, "pssh: Invalid signal %i\n", signal);
        }
        else if ( errno == ESRCH ) {
            fprintf(stderr, "pssh: PID %i does not exist\n", pid);
        }
        else {
            fprintf(stderr, "pssh: ERROR sending signal %i to pid %i. ERRNO=%i\n", signal, pid, errno);        
        }
        return -1;
    }
    else {
        if ( signal == 0 ) {
            printf("pssh: PID %i exists and is able to receive signals\n", pid);
        }
        else {
            // printf("Sent signal %d to pid %ld\n", signal, pid);
        }
        return 0;
    }
}

int str_to_sig(char* str)
{
    if ( !strcmp(str, "SIGHUP") )
        return 1;
    else if ( !strcmp(str, "SIGINT") )
        return 2;
    else if ( !strcmp(str, "SIGQUIT") )
        return 3;
    else if ( !strcmp(str, "SIGILL") )
        return 4;
    else if ( !strcmp(str, "SIGTRAP") )
        return 5;
    else if ( !strcmp(str, "SIGABRT") )
        return 6;
    else if ( !strcmp(str, "SIGBUS") )
        return 7;
    else if ( !strcmp(str, "SIGFPE") )
        return 8;
    else if ( !strcmp(str, "SIGKILL") )
        return 9;
    else if ( !strcmp(str, "SIGUSR1") )
        return 10;
    else if ( !strcmp(str, "SIGSEGV") )
        return 11;
    else if ( !strcmp(str, "SIGUSR2") )
        return 12;
    else if ( !strcmp(str, "SIGPIPE") )
        return 13;
    else if ( !strcmp(str, "SIGALRM") )
        return 14;
    else if ( !strcmp(str, "SIGTERM") )
        return 15;
    else if ( !strcmp(str, "SIGSTKFLT") )
        return 16;
    else if ( !strcmp(str, "SIGCHLD") )
        return 17;
    else if ( !strcmp(str, "SIGCONT") )
        return 18;
    else if ( !strcmp(str, "SIGSTOP") )
        return 19;
    else if ( !strcmp(str, "SIGTSTP") )
        return 20;
    else if ( !strcmp(str, "SIGTTIN") )
        return 21;
    else if ( !strcmp(str, "SIGTTOU") )
        return 22;
    else if ( !strcmp(str, "SIGURG") )
        return 23;
    else if ( !strcmp(str, "SIGXCPU") )
        return 24;
    else if ( !strcmp(str, "SIGXFSZ") )
        return 25;
    else if ( !strcmp(str, "SIGVTALRM") )
        return 26;
    else if ( !strcmp(str, "SIGPROF") )
        return 27;
    else if ( !strcmp(str, "SIGWINCH") )
        return 28;
    else if ( !strcmp(str, "SIGIO") )
        return 29;
    else if ( !strcmp(str, "SIGPWR") )
        return 30;
    else if ( !strcmp(str, "SIGSYS") )
        return 31;
    else
        return -1;
}
