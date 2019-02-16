#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <dirent.h>
#include <sys/stat.h>

#include "builtin.h"
#include "parse.h"

static char* builtin[] = {
    "exit",   /* exits the shell */
    "which",  /* displays full path to command */
    NULL
};


int is_builtin (char* cmd)
{
    int i;

    for (i=0; builtin[i]; i++) {
        if (!strcmp (cmd, builtin[i]))
            return 1;
    }

    return 0;
}

/* Return 1 if program is found in dir, 0 if not. Print out the absolute path to that program */
int find_program(char* curdir, char* dir, char* program)
{
    DIR *dp;
    struct dirent *entry;
    struct stat statbuf;

    int ret = 0;

    if ( (dp = opendir(dir)) == NULL ) {
        return ret;
    }
    
    chdir(dir);

    while ( (entry = readdir(dp)) != NULL ) {

        if ( !strcmp(entry->d_name, program) ) { // if it's a match
            if ( stat(entry->d_name, &statbuf) == 0 && statbuf.st_mode & S_IXUSR) { // if the match is executable by the user
                printf("%s/%s\n", dir, entry->d_name);
                ret = 1;
                break;
            }
        }
    }
    chdir(curdir);
    closedir(dp);

    return ret;
}


/* Scan all $PATH to see if the input program is there, 
if not, check if `program`, as a path to a file, is executable by the user
and print out the `program`'s absolute path */
void builtin_which (char* program)
{
    if ( is_builtin(program) ) {
        printf("%s: shell built-in command\n", program);
        exit(EXIT_SUCCESS);
    }

    char* path_env = getenv("PATH");    // get $PATH
    char* paths[1000];                  // array of 1000 pointers to paths in $PATH
    paths[0] = strtok(path_env, ":");   // get the first path
    
    int i, found;
    char* path;
    char curdir[PATH_MAX];
    getcwd(curdir, PATH_MAX);
    
    i = 0;
    while (paths[i] != NULL) {
        path = paths[i];
        found = find_program(curdir, path, program);
        if ( found )
            break;
        else
            paths[++i] = strtok(NULL, ":"); // Go to the next path
    }

    if ( !found ) { // if program is not in path, check if program is a path to a file
        char abs_path[PATH_MAX];
        struct stat statbuf;

            if ( stat(program, &statbuf) == 0 && statbuf.st_mode & S_IXUSR) { // if the match is executable by the user
                realpath(program, abs_path);
                printf("%s\n", abs_path);   // print out the file's absolute path
            }

    }

    exit(EXIT_SUCCESS);
}

void builtin_execute (char* cmd, char** argv)
{
    if (!strcmp (cmd, "exit")) {
        exit (EXIT_SUCCESS);
    }
    else if (!strcmp (cmd, "which")) {
        builtin_which(argv[1]);
    }
    else {
        printf ("pssh: builtin command: %s (not implemented!)\n", cmd);
    }
}
