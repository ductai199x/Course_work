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
void sigttin_handler(int sig);
void sigchild_handler(int sig);
void sigint_handler(int sig);
void sigtstp_handler(int sig);
void sigquit_handler(int sig);

int child_stopped = 0;
job_t* J_term;
job_t* J_stop;
job_t* J_cont;

void sigchild_handler(int sig)
{
    pid_t child;
    int status;
    JobStatus jstatus;

    char prnt[1000];

    while( (child = waitpid(-1, &status, WNOHANG | WUNTRACED | WCONTINUED)) > 0 ) {
        if ( WIFSTOPPED(status) ) {
            if ( (get_job(child)) != NULL ) {
                J_stop = get_job(child);
            }                
            if ( J_stop ) {
                child_stopped++;
                if ( J_stop->npids == child_stopped) {
                    J_stop->status = SUSPENDED;
                    view_job(J_stop, prnt);
                    safe_print(prnt);
                    J_stop->status = STOPPED;
                    
                    child_stopped = 0;
                    set_fg_pgid(getpgrp());
                    J_stop = NULL;
                }
            }        
        }
        else if ( WIFCONTINUED(status) ) {
            if ( child == getpgid(child) ) {
                if ( (J_cont = get_job(getpgid(child))) != NULL ) {
                    if ( J_cont-> status == STOPPED ) {
                        J_cont->status = CONTINUED;
                    }
                    view_job(J_cont, prnt);
                    safe_print(prnt);
                    J_cont->status = BG;
                }
            }
            J_cont = NULL;
        }
        else {
            if ( get_job(child) != NULL ) {
                J_term = get_job(child);
            } else {
                // set_fg_pgid(getpgrp());
            }                
            if ( J_term ) {
                remove_pid_from_job(J_term, child);
                if ( J_term->active_pids == 0 ) {
                    remove_job(J_term);
                    jstatus = J_term->status;
                    if ( jstatus == BG || jstatus == STOPPED || jstatus == KILLED ) {
                        if ( jstatus == BG ) {
                            safe_print("\n");
                            J_term->status = TERM;
                            view_job(J_term, prnt);
                            safe_print(prnt);
                        } else {
                            J_term->status = TERM;
                            view_job(J_term, prnt);
                            safe_print(prnt);
                        }
                    }
                    if ( !(jstatus == ADMIN ? 1 : 0) ) {
                        set_fg_pgid(getpgrp());
                    }
                    free(J_term);
                    J_term = NULL;
                }        
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

void sigttin_handler(int sig)
{
    while( tcgetpgrp(STDIN_FILENO) != getpid() ) {
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

void execute_command(char* cmd, char** argv, char* infile, char* outfile, int* oldfd, int* newfd, pid_t* pid_arr, int index, int background)
{
    pid_t pid;
    pid = fork();

    setpgid(pid, pid_arr[0]);
    pid_arr[index] = pid;
    
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

        signal (SIGINT, SIG_DFL);
        signal (SIGTSTP, SIG_DFL);
        signal (SIGQUIT, SIG_DFL);
        signal (SIGCHLD, SIG_DFL);

        if ( is_builtin(cmd) ) {
            builtin_execute(cmd, argv);
            exit(EXIT_SUCCESS);
        } else {
            execvp(cmd, argv);
        }
    } else {                    // parent process
        char prnt[10];
        if ( index == 0 ) {
            if ( !background ) {
                set_fg_pgid(pid);
            }
        } 
        
        if ( oldfd ) {          // closing the old pipe
            close(oldfd[0]);
            close(oldfd[1]);
        }
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
    pid_t* pid_arr = (pid_t*)malloc(num_tasks*sizeof(pid_t));

    JobStatus status = P->background ? BG : FG;;
    job_t* J = malloc(sizeof(J));   // free later

    if ( (J = add_job(P, pid_arr, status)) == NULL ) {
        fprintf(stderr, "Failed to add new job\n");
    }

    J->npids = num_tasks;
    J->active_pids = num_tasks;

    for ( t = 0; t < num_tasks; t++ ) {
        if ( command_found (P->tasks[t].cmd) || is_builtin (P->tasks[t].cmd) ) 
        {
            if ( !strcmp(P->tasks[t].cmd, "exit") )
                exit(EXIT_SUCCESS);
            
            if ( !strcmp(P->tasks[t].cmd, "fg") || !strcmp(P->tasks[t].cmd, "kill") ) {
                J->status = ADMIN;
            }

            if ( t < num_tasks-1 ) {    // if the current task is not the last one, create pipe
                if ( pipe(newfd) == -1 ) {
                    fprintf(stderr, "Create pipe failed\n");
                    exit(EXIT_FAILURE);
                }
            }

            if ( num_tasks > 1 ) {      // if there is more than one task, do things with pipes
                if ( t == 0 ) {
                    execute_command(P->tasks[t].cmd, P->tasks[t].argv, P->infile, NULL, NULL, newfd, pid_arr, t, P->background);
                } else if ( t > 0 && t < num_tasks-1 ) {
                    execute_command(P->tasks[t].cmd, P->tasks[t].argv, NULL, NULL, oldfd, newfd, pid_arr, t, P->background);
                } else {
                    execute_command(P->tasks[t].cmd, P->tasks[t].argv, NULL, P->outfile, oldfd, NULL, pid_arr, t, P->background);
                }

                memcpy(oldfd, newfd, sizeof(newfd));
            } else {    // if there is only one tasks, dont care about pipes
                execute_command(P->tasks[t].cmd, P->tasks[t].argv, P->infile, P->outfile, NULL, NULL, pid_arr, t, P->background);
            }

            if ( P->background ) {
                char prnt[10];
                sprintf(prnt, "%i ", pid_arr[t]);
                safe_print(prnt);
            }
            
        }
        else 
        {
            remove_job(J);
            printf ("pssh: command not found: %s\n", P->tasks[t].cmd);
            break;
        }
    }
    if ( P->background ) {
        safe_print("\n");
    }
}


int main (int argc, char** argv)
{
    char* cmdline;
    Parse* P;

    print_banner();
    
    signal(SIGCHLD, sigchild_handler);
    signal(SIGTTOU, sigttou_handler);
    signal(SIGTTIN, sigttin_handler);
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
