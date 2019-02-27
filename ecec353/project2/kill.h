#ifndef _kill_h_
#define _kill_h_

#include <signal.h>

typedef struct killsig {
    int signal;
    int* pids;
    int* jobs;
} killsig_t;

killsig_t* parse_kill_args(char** argv);
int str_to_sig(char* str);
void print_usage();
int send_signal(int signal, int pid);

#endif /* _kill_h_ */
