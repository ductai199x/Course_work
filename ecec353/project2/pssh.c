#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <readline/readline.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <signal.h>
#include <errno.h>

#include "builtin.h"
#include "parse.h"
#include "job.h"
#include "kill.h"
#include "helper.h"

/*******************************************
 * Set to 1 to view the command line parse *
 *******************************************/
#define DEBUG_PARSE 0

void print_banner ()
{
    printf ("                    ________   \n");
    printf ("_________________________  /_  \n");
    printf ("___  __ \\_  ___/_  ___/_  __ \\ \n");
    printf ("__  /_/ /(__  )_(__  )_  / / / \n");
    printf ("_  .___//____/ /____/ /_/ /_/  \n");
    printf ("/_/ Type 'exit' or ctrl+c to quit\n\n");
}


/* returns a string for building the prompt
 *
 * Note:
 *   If you modify this function to return a string on the heap,
 *   be sure to free() it later when appropirate!  */
void build_prompt (char* curr_path)
{
    getcwd(curr_path, PATH_MAX);
    strncat(curr_path, "$ ", PATH_MAX+2);
}


/* return true if command is found, either:
 *   - a valid fully qualified path was supplied to an existing file
 *   - the executable file was found in the system's PATH
 * false is returned otherwise */
static int command_found (const char* cmd)
{
    char* dir;
    char* tmp;
    char* PATH;
    char* state;
    char probe[PATH_MAX];

    int ret = 0;

    if (access (cmd, X_OK) == 0)
        return 1;
    
    if ( !strcmp(cmd, "exit") )
        return 1;    

    PATH = strdup (getenv("PATH"));

    for (tmp=PATH; ; tmp=NULL) {
        dir = strtok_r (tmp, ":", &state);
        if (!dir)
            break;

        strncpy (probe, dir, PATH_MAX);
        strncat (probe, "/", PATH_MAX-1);
        strncat (probe, cmd, PATH_MAX);

        if (access (probe, X_OK) == 0) {
            ret = 1;
            break;
        }
    }

    free (PATH);
    return ret;
}

// PROJECT 2

void sigttou_handler(int sig);
void sigchild_handler(int sig);
void sigint_handler(int sig);
void sigtstp_handler(int sig);
void sigquit_handler(int sig);

sig_atomic_t child_term = 0;

void sigchild_handler(int sig)
{
    pid_t child;
    int status;
    job_t* J;
    char prnt[1000];
    int admin = 0;
    
    while( (child = waitpid(-1, &status, WNOHANG | WUNTRACED | WCONTINUED)) > 0 ) {
        if ( WIFSTOPPED(status) ) {
            child_term = child_term ? child_term-1 : 0;
            
            if ( (J = get_job(child)) != NULL ) {
                J->status = SUSPENDED;
                view_job(J, prnt);
                safe_print(prnt);
                J->status = STOPPED;
            }
            if ( child_term == 0 ) {
                set_fg_pgid(getpgrp()); 
            }
        }
        else if ( WIFCONTINUED(status) ) {
            if ( child == getpgid(child) ) {
                if ( (J = get_job(getpgid(child))) != NULL ) {
                    if ( J-> status == STOPPED ) {
                        J->status = CONTINUED;
                    }
                    view_job(J, prnt);
                    safe_print(prnt);
                    J->status = BG;
                }
            }
        }
        else {
            child_term = child_term ? child_term-1 : 0;
            
            // printf("child_term: %i\n", child_term);
            if ( (J = remove_job(child)) != NULL ) {
                if ( J->status == BG || J->status == STOPPED || J->status == KILLED ) {
                    if ( J->status != KILLED ) {
                        J->status = TERM;
                    }
                    view_job(J, prnt);
                    safe_print(prnt);
                }
                admin = J->status == ADMIN ? 1 : 0;
            }
            if ( child_term == 0 ) {
                if ( !admin ) {
                    set_fg_pgid(getpgrp());
                }
                admin = 0;
            }
            continue;
        }
    }
}

void sigttou_handler(int sig)
{
    while( tcgetpgrp(STDOUT_FILENO) != getpid() ) {
        pause();
    }
}

void sigint_handler(int sig)
{
    pid_t curr_tc = tcgetpgrp(STDOUT_FILENO);
    if ( curr_tc != getpgrp() ) {
        kill(-curr_tc, SIGINT);
    }
}

void sigtstp_handler(int sig)
{
    pid_t curr_tc = tcgetpgrp(STDOUT_FILENO);
    if ( curr_tc != getpgrp() ) {
        kill(-curr_tc, SIGTSTP);
    }
}

void sigquit_handler(int sig)
{
    pid_t curr_tc = tcgetpgrp(STDOUT_FILENO);
    if ( curr_tc != getpgrp() ) {
        kill(-curr_tc, SIGQUIT);
    }
}

// PROJECT 2

void execute_command(char* cmd, char** argv, char* infile, char* outfile, int* oldfd, int* newfd, pid_t* pgid, int lead, int background)
{
    pid_t pid;
    pid = fork();
    
    if ( !background ) {
        child_term++;
    }

    setpgid(pid, *pgid);
    if ( lead )
        *pgid = pid;
    
    if ( pid < 0 ) {
        fprintf(stderr, "Failed to fork()\n");
        exit(EXIT_FAILURE);
    } else if ( pid == 0 ) {    // child process
        if ( oldfd ) {          // old pipe
            dup2(oldfd[0], STDIN_FILENO);
            close(oldfd[0]);
            close(oldfd[1]);
        }
        
        if ( newfd ) {          // new pipe
            close(newfd[0]);
            dup2(newfd[1], STDOUT_FILENO);
            close(newfd[1]);
        }

        if ( infile ) {         // in-file
            int if_fd = open(infile, O_RDONLY);
            dup2(if_fd, STDIN_FILENO);
            close(if_fd);
        }

        if ( outfile ) {        // out-file
            int of_fd = open(outfile, O_WRONLY | O_CREAT | O_TRUNC, 0644);
            dup2(of_fd, STDOUT_FILENO);
            close(of_fd);
        }

        if ( is_builtin(cmd) ) {
            builtin_execute(cmd, argv);
            exit(EXIT_SUCCESS);
        } else {
            execvp(cmd, argv);
        }
    } else {                    // parent process
        if ( oldfd ) {          // closing the old pipe
            close(oldfd[0]);
            close(oldfd[1]);
        }

        if ( lead && !background ) {
            set_fg_pgid(pid);
        }
        // printf("bg: %i - pgid: %i\n", background, tcgetpgrp(STDOUT_FILENO));
    }
}


/* Called upon receiving a successful parse.
 * This function is responsible for cycling through the
 * tasks, and forking, executing, etc as necessary to get
 * the job done! */
void execute_tasks (Parse* P)
{
    unsigned int t;
    
    int oldfd[2];
    int newfd[2];

    int num_tasks = P->ntasks;  // just for convenience sake
    pid_t pgid = 0;

    JobStatus status;

    for ( t = 0; t < num_tasks; t++ ) {
        if ( command_found (P->tasks[t].cmd) || is_builtin (P->tasks[t].cmd) ) {
            if ( !strcmp(P->tasks[t].cmd, "exit") )
                exit(EXIT_SUCCESS);

            if ( t < num_tasks-1 ) {    // if the current task is not the last one, create pipe
                if ( pipe(newfd) == -1 ) {
                    fprintf(stderr, "Create pipe failed\n");
                    exit(EXIT_FAILURE);
                }
            }

            if ( num_tasks > 1 ) {      // if there is more than one task, do things with pipes
                if ( t == 0 ) {
                    execute_command(P->tasks[t].cmd, P->tasks[t].argv, P->infile, NULL, NULL, newfd, &pgid, 1, P->background);
                    memcpy(oldfd, newfd, sizeof(newfd));
                    if ( !strcmp(P->tasks[t].cmd, "fg") ) {
                        status = ADMIN;
                    } else {
                        status = P->background ? BG : FG;
                    }
                    if ( add_job(P, pgid, status) == -1 ) {
                        fprintf(stderr, "Failed to add new job\n");
                    }
                } else if ( t > 0 && t < num_tasks-1 ) {
                    execute_command(P->tasks[t].cmd, P->tasks[t].argv, NULL, NULL, oldfd, newfd, &pgid, 0, P->background);
                    memcpy(oldfd, newfd, sizeof(newfd));
                } else {
                    execute_command(P->tasks[t].cmd, P->tasks[t].argv, NULL, P->outfile, oldfd, NULL, &pgid, 0, P->background);
                }
            } else {    // if there is only one tasks, dont care about pipes
                execute_command(P->tasks[t].cmd, P->tasks[t].argv, P->infile, P->outfile, NULL, NULL, &pgid, 1, P->background);
                if ( !strcmp(P->tasks[t].cmd, "fg") ) {
                    status = ADMIN;
                } else {
                    status = P->background ? BG : FG;
                }
                if ( add_job(P, pgid, status) == -1 ) {
                    fprintf(stderr, "Failed to add new job\n");
                }
            }
        }
        else {
            printf ("pssh: command not found: %s\n", P->tasks[t].cmd);
            break;
        }
    }
}


int main (int argc, char** argv)
{
    char* cmdline;
    Parse* P;

    print_banner();
    
    signal(SIGCHLD, sigchild_handler);
    signal(SIGTTOU, sigttou_handler);
    signal(SIGINT, sigint_handler);
    signal(SIGTSTP, sigtstp_handler);
    signal(SIGQUIT, sigquit_handler);

    char curr_path[PATH_MAX];

    while (1) {
        while ( tcgetpgrp(STDOUT_FILENO)!=getpid() ) { 
            pause(); 
        }
        build_prompt(curr_path);
        cmdline = readline (curr_path);
        if (!cmdline)       /* EOF (ex: ctrl-d) */
            exit (EXIT_SUCCESS);

        P = parse_cmdline (cmdline);
        if (!P)
            goto next;

        if (P->invalid_syntax) {
            printf ("pssh: invalid syntax\n");
            goto next;
        }

#if DEBUG_PARSE
        parse_debug (P);
#endif

        execute_tasks (P);

    next:
        parse_destroy (&P);
        free(cmdline);
    }
}
