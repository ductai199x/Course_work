#ifndef _builtin_h_
#define _builtin_h_

#include "parse.h"
#include "job.h"

int is_builtin (char* cmd);
void builtin_execute (char* cmd, char** argv);
void builtin_which (char* program);
void builtin_jobs ();
void builtin_fg (char** argv);
void builtin_bg (char** argv);

#endif /* _builtin_h_ */
