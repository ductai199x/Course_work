#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "job.h"
#include "parse.h"

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

int add_job(Parse* p, pid_t pgid, JobStatus status)
{
    if ( job_num == JOB_MAX-1 ) return -1;

    job_t* J = malloc(sizeof(*J));
    J = parse_job(p);
    J->pgid = pgid;
    J->status = status;
    J->num = job_num;

    job_list[job_num] = J;
    job_num++;
    
    return job_num;
}

job_t* parse_job(Parse* p)
{
    job_t* J = malloc(sizeof(J));
    J = new_job();
    J->name = malloc(sizeof(p->raw_cmd));
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
    for ( i = JOB_MIN; i < job_num; i++ ) {
        if ( job_list[i]->pgid == pgid ) {
            ret = job_list[i];
            break;
        }
    }
    
    if ( i < job_num ) {
        return ret;
    } else {
        return NULL;
    }
}

job_t* remove_job(pid_t pgid)
{
    int i, j;
    job_t* ret = malloc(sizeof(ret));
    for ( i = JOB_MIN; i < job_num; i++ ) {
        if ( job_list[i]->pgid == pgid ) {
            ret = job_list[i];
            break;
        }
    }
    if ( i < job_num ) {
        for( j = i; j < job_num; j++ ) {
            job_list[j] = job_list[j+1];
        }
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

    int i;
    for ( i = job_id; i < job_num; i++ ) {
        job_list[i] = job_list[i+1];
    }
    return ret;
}

void view_all_jobs()
{
    int i;
    for ( i = JOB_MIN; i < job_num; i++ ) {
        if ( job_list[i] ) {
            printf("[%i] + %s\t%s\n", i, get_str_status(job_list[i]->status), job_list[i]->name);
        }
    }
}

char* get_str_status(JobStatus s)
{
    if ( s == 0 )
        return "Stopped";
    else if ( s == 1 )
        return "Terminated";
    else if ( s == 2 )
        return "Running-BG";
    else if ( s == 3 )
        return "Running-FG";
    else
        return "Unknown";
}

void view_job(job_t* J, char* prnt)
{
    sprintf(prnt, "[%i] + %s\t%s\n", J->num, get_str_status(J->status), J->name);
}

