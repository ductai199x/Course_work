#ifndef __BLISS_TABLE_HH__
#define __BLISS_TABLE_HH__

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

typedef struct BLISS_table
{
    unsigned core_id;
    unsigned num_req_served;
    unsigned blacklist_threshold;
    int* blacklisted_ids;
    uint64_t clearing_interval;
    uint64_t addr_just_served;
}BLISS_table;






#endif