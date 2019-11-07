#ifndef _builtin_h_
#define _builtin_h_

#include "parse.h"

int is_builtin (char* cmd);
void builtin_execute (char* cmd, char** argv);
void builtin_which (char* program);

#endif /* _builtin_h_ */
