#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <signal.h>

#include "fg.h"
#include "job.h"
#include "helper.h"

void print_usage_fg()
{
    printf("Usage: fg %%<job number>\n");
}

fg_t* parse_fg_args(char** argv)
{
    fg_t* f = malloc(sizeof(f));
    f->pids = malloc(1*sizeof(int));
    f->jobs = malloc(1*sizeof(int));
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
        print_usage_fg();
        return NULL;
    }

    int i, k;
    char *start, *end, *arg;
    i = 1;
    for ( k = 0; k < strlen(argv[i]); k++ ) {
        if ( argv[i][k] != '%' ) break;
    }
    start = (k + argv[i]);
    end = (strlen(argv[i]) + argv[i]);
    arg = strndup(start, end - start);
    arg[end - start] = '\0';
    if ( check_int(arg) ) {
        *j = atoi(arg);
        j++;
    } else {
        fprintf(stderr, "pssh: fg: invalid job number [job number]\n");
        return NULL;
    }

    *p = -1;
    *j = -1;
    f->pids = pids;
    f->jobs = jobs;

    return f;
}



