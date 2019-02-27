#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <signal.h>

#include "bg.h"
#include "job.h"
#include "helper.h"

void print_usage_bg()
{
    printf("Usage: bg %%<job number>\n");
}

bg_t* parse_bg_args(char** argv)
{
    bg_t* f = malloc(sizeof(f));
    int *pids = (int*)malloc(1 * sizeof(int));
    int *jobs = (int*)malloc(1 * sizeof(int));
    int *p, *j;
    p = pids;
    j = jobs;

    int argc = 0;
    while ( argv[argc] ) {
        argc++;
    }

    if ( argc < 2 ) {
        print_usage_bg();
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



