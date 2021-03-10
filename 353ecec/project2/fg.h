#ifndef _fg_h_
#define _fg_h_

#include "helper.h"

typedef struct fg {
    int* pids;
    int* jobs;
} fg_t;

fg_t* parse_fg_args(char** argv);
int check_int(char* str);

#endif /* _fg_h_ */
