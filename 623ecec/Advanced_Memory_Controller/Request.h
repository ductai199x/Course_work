#ifndef __REQUEST_HH__
#define __REQUEST_HH__

#define __STDC_FORMAT_MACROS
#include <inttypes.h> // uint64_t

typedef enum Request_Type{READ, WRITE}Request_Type;

// Instruction Format
typedef struct Request
{
    int core_id; // The core-id/app-id sends the request.

    Request_Type req_type;

    uint64_t memory_address;

    /* Decoding Info */
    int channel_id;
    int bank_id; // Which bank it targets to.
    uint64_t queued_time;

}Request;

#endif
