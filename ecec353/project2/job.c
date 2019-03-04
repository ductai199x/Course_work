#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "job.h"
#include "parse.h"
#include "helper.h"

#define JOB_MAX 100
#define JOB_MIN 1
//Job Struct:
//  int num;
//  char* name;
//  unsigned int npids;
//  pid_t pgid;
//  JobStatus status;

job_t* job_list[JOB_MAX];
int job_num = JOB_MIN;
int last_added_job_num = JOB_MIN;
int highest_job_num = JOB_MIN;

int get_lowest_unused_num()
{
    int i = 1;
    while ( job_list[i] ) {
        if ( job_list[i] == NULL ) {
            break;
        }
        i++;
    }

    return i;
}

int add_job(Parse* p, pid_t pgid, JobStatus status)
{
    if ( job_num == JOB_MAX-1 ) return -1;

    job_t* J = malloc(sizeof(*J));
    J = parse_job(p);

    int n = get_lowest_unused_num();
    J->pgid = pgid;
    J->status = status;
    J->num = n;

    last_added_job_num = n;

    if ( n > highest_job_num ) {
        highest_job_num = n;
    }

    job_list[n] = J;
    if ( status == BG ) {
        char prnt[10];
        sprintf(prnt, "[%i] %i\n", J->num, J->pgid);
        safe_print(prnt);
    }
    job_num++;
    
    return n;
}

job_t* parse_job(Parse* p)
{
    job_t* J = malloc(sizeof(J));
    J = new_job();
    J->name = malloc(strlen(p->raw_cmd) + 1);
    strcpy(J->name, p->raw_cmd);
    J->npids = p->ntasks;

    return J;
}

job_t* new_job()
{
    job_t* J = malloc (sizeof(*J));

    J->num = -1;
    J->name = NULL;
    J->npids = -1;
    J->pgid = -1;
    J->status = UNKNOWN;

    return J;
}

job_t* get_last_job()
{
    if ( job_num == 1 ) return NULL;
    
    return job_list[job_num-1];
}

job_t* pop_last_job()
{
    if ( job_num == 1 ) return NULL;
   
    job_t* ret = malloc(sizeof(ret));
    ret = job_list[job_num-1];
    job_list[job_num-1] = NULL;
    job_num--;
    
    return ret;
}

job_t* get_job(pid_t pgid)
{
    int i;
    job_t* ret = malloc(sizeof(ret));
    for ( i = JOB_MIN; i <= highest_job_num; i++ ) {
        if ( job_list[i] ) {
            if ( job_list[i]->pgid == pgid ) {
                ret = job_list[i];
                break;
            }
        }
    }
    
    if ( i <= highest_job_num ) {
        return ret;
    } else {
        return NULL;
    }
}

job_t* get_job_with_id(int job_id)
{
    if ( job_num < 1 || job_id < 1 ) return NULL;
    if ( job_id >= job_num ) return NULL;
    job_t* ret = malloc(sizeof(ret));
    ret = job_list[job_id];

    return ret;
}

job_t* remove_job(pid_t pgid)
{
    int i;
    job_t* ret = malloc(sizeof(ret));
    for ( i = JOB_MIN; i <= highest_job_num; i++ ) {
        if ( job_list[i] ) {
            if ( job_list[i]->pgid == pgid ) {
                ret = job_list[i];
                break;
            }
        }
    }
    if ( i <= highest_job_num ) {
        job_list[i] = NULL;
        job_num--;
        return ret;
    } else {
        return NULL;
    }
}

job_t* remove_job_with_id(int job_id)
{
    if ( job_num < 1 || job_id < 1 ) return NULL;
    if ( job_id >= job_num ) return NULL;
    job_t* ret = malloc(sizeof(ret));
    ret = job_list[job_id];
    
    job_list[job_id] = NULL;

    return ret;
}

void view_all_jobs()
{
    int i;
    char prnt[1000];
    for ( i = JOB_MIN; i < job_num; i++ ) {
        if ( job_list[i] ) {
            sprintf(prnt, "[%i] + %s\t\t%s\n", i, get_str_status(job_list[i]->status), job_list[i]->name);
            safe_print(prnt);
            memset(prnt, 0, 1000);
        }
    }
}

char* get_str_status(JobStatus s)
{
    if ( s == 0 )
        return "stopped";
    else if ( s == 1 )
        return "terminated";
    else if ( s == 2 )
        return "running";
    else if ( s == 3 )
        return "running";
    else if ( s == 4 )
        return "done";
    else if ( s == 5 )
        return "continued";
    else if ( s == 6 )
        return "suspended";
    else
        return "Unknown";
}

void view_job(job_t* J, char* prnt)
{
    sprintf(prnt, "[%i] + %s\t\t%s\n", J->num, get_str_status(J->status), J->name);
}

