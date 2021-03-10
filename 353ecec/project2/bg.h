#ifndef _bg_h_
#define _bg_h_

#include "helper.h"

typedef struct bg {
    int* pids;
    int* jobs;
} bg_t;

bg_t* parse_bg_args(char** argv);
int check_int(char* str);

#endif /* _bg_h_ */
