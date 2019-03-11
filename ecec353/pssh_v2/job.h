#ifndef _job_mgmt_
#define _job_mgmt_

#include <signal.h>

#include "parse.h"

typedef enum {
        STOPPED,
        TERM,
        BG,
        FG,
        KILLED,
        CONTINUED,
        SUSPENDED,
        UNKNOWN,
        ADMIN,
} JobStatus;

typedef struct Job {
        int num;
        char* name;
        pid_t* pid_arr;
        unsigned int npids;
        unsigned int active_pids;
        pid_t pgid;
        JobStatus status;
} job_t;

job_t* add_job(Parse* p, pid_t* pid_arr, JobStatus status);
job_t* pop_last_job();
job_t* get_last_job();
job_t* get_job(pid_t pgid);
job_t* get_job_with_id(int job_id);
int remove_pid_from_job(job_t* J, int pid);
int remove_job(job_t* J);
job_t* remove_job_with_id(int job_id);
char* get_str_status(JobStatus s);
void view_all_jobs();
void view_job(job_t* J, char* prnt);

job_t* parse_job(Parse* p);
job_t* new_job();

#endif /* _job_mgmt_ */
