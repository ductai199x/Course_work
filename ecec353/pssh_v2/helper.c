#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>

#include "helper.h"

void set_fg_pgid(pid_t pgid)
{
    void (*old)(int);
    old = signal(SIGTTOU, SIG_IGN);
    tcsetpgrp(STDIN_FILENO, pgid);
    tcsetpgrp(STDOUT_FILENO, pgid);
    // printf("tc set to: %i\n", tcgetpgrp(STDOUT_FILENO));
    signal(SIGTTOU, old);
}

void safe_print(char* str)
{
    pid_t fg_pgid;
    fg_pgid = tcgetpgrp(STDOUT_FILENO);
    set_fg_pgid(getpgrp());
    printf("%s", str);
    set_fg_pgid(fg_pgid);
}

int check_int(char* str)
{
   int i;
   for ( i = 0; i < strlen(str); i++ ) {
       if ( !isdigit(str[i]) )
           return 0;
   }
   return 1;
}

