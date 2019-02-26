#ifndef _job_mgmt_
#define _job_mgmt_

#include <signal.h>

#include "parse.h"

typedef enum {
        STOPPED,
        TERM,
        BG,
        FG,
        UNKNOWN,
} JobStatus;

typedef struct Job {
        int num;
        char* name;
        // pid_t* pids;
        unsigned int npids;
        pid_t pgid;
        JobStatus status;
        struct Job *next_job;
        struct Job *prev_job;
} job_t;

int add_job(Parse* p, pid_t pgid, JobStatus status);
job_t* pop_last_job();
job_t* get_last_job();
job_t* get_job(pid_t pgid);
job_t* remove_job(pid_t pgid);
job_t* remove_job_with_id(int job_id);
char* get_str_status(JobStatus s);
void view_all_jobs();
void view_job(job_t* J, char* prnt);

job_t* parse_job(Parse* p);
job_t* new_job();

#endif /* _job_mgmt_ */
