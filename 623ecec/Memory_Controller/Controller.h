#ifndef __CONTROLLER_HH__
#define __CONTROLLER_HH__

#include "Bank.h"
#include "Queue.h"

// Bank
extern void initBank(Bank *bank);

// Queue operations
extern Queue* initQueue();
extern void pushToQueue(Queue *q, Request *req);
extern void migrateToQueue(Queue *q, Node *_node);
extern void deleteNode(Queue *q, Node *node);

// // CONSTANTS
// static unsigned MAX_WAITING_QUEUE_SIZE = 64;
// static unsigned BLOCK_SIZE = 128; // cache block size
// static unsigned NUM_OF_BANKS = 2; // number of banks

// // DRAM Timings
// static unsigned nclks_read = 53;
// static unsigned nclks_write = 53;

// PCM Timings
// static unsigned nclks_read = 57;
// static unsigned nclks_write = 162;

// Controller definition
typedef struct Controller
{
    // The memory controller needs to maintain records of all bank's status
    Bank *bank_status;

    // Current memory clock
    uint64_t cur_clk;

    // A queue contains all the requests that are waiting to be issued.
    Queue *waiting_queue;

    // A queue contains all the requests that have already been issued but are waiting to complete.
    Queue *pending_queue;

    /* For decoding */
    unsigned bank_shift;
    uint64_t bank_mask;

    // Access Latency
    uint64_t access_time;

    // Bank-Conflicts
    uint64_t bank_conficts;

    // Configs
    unsigned nclks_read;
    unsigned nclks_write;
    unsigned max_waiting_queue_size; // Size of a cache line (in Bytes)
    unsigned block_size; // Size of a cache (in KB)
    unsigned num_of_banks;
    bool is_frfcfs;

}Controller;

// Controller configs
typedef struct Controller_Config
{
    unsigned nclks_read;
    unsigned nclks_write;
    unsigned max_waiting_queue_size; // Size of a cache line (in Bytes)
    unsigned block_size; // Size of a cache (in KB)
    unsigned num_of_banks;
    bool is_frfcfs;
}Controller_Config;

Controller *initController(Controller_Config* config)
{

    Controller *controller = (Controller *)malloc(sizeof(Controller));

    controller->nclks_read = config->nclks_read;
    controller->nclks_write = config->nclks_write;
    controller->max_waiting_queue_size = config->max_waiting_queue_size; // Size of a cache line (in Bytes)
    controller->block_size = config->block_size; // Size of a cache (in KB)
    controller->num_of_banks = config->num_of_banks;
    controller->is_frfcfs = config->is_frfcfs;


    controller->bank_status = (Bank *)malloc(controller->num_of_banks * sizeof(Bank));
    for (int i = 0; i < controller->num_of_banks; i++)
    {
        initBank(&((controller->bank_status)[i]));
    }
    controller->cur_clk = 0;

    controller->waiting_queue = initQueue();
    controller->pending_queue = initQueue();

    controller->bank_shift = log2(controller->block_size);
    controller->bank_mask = (uint64_t)controller->num_of_banks - (uint64_t)1;

    controller->access_time = 0;
    controller->bank_conficts = 0;

    return controller;
}

unsigned ongoingPendingRequests(Controller *controller)
{
    unsigned num_requests_left = controller->waiting_queue->size + 
                                 controller->pending_queue->size;

    return num_requests_left;
}

bool send(Controller *controller, Request *req)
{
    if (controller->waiting_queue->size == controller->max_waiting_queue_size)
    {
        return false;
    }

    // Decode the memory address
    req->bank_id = ((req->memory_address) >> controller->bank_shift) & controller->bank_mask;
    req->queued_time = controller->cur_clk;
    // Push to queue
    pushToQueue(controller->waiting_queue, req);

    return true;
}

void tick(Controller *controller, uint64_t *conflict_req)
{
    // Step one, update system stats
    ++(controller->cur_clk);
    for (int i = 0; i < controller->num_of_banks; i++)
    {
        ++(controller->bank_status)[i].cur_clk;
    }

    // Step two, serve pending requests
    if (controller->pending_queue->size)
    {
        Node *first = controller->pending_queue->first;
        if (first->end_exe <= controller->cur_clk)
        {
            deleteNode(controller->pending_queue, first);
        }
    }

    // Step three, find a request to schedule
    if (controller->waiting_queue->size)
    {
        
        // Implementation One - FCFS
        Node *first = controller->waiting_queue->first;
        Node *new_first = controller->waiting_queue->first;
        int is_bank_confl = 1;
        int is_incr_bank_confl = 0;

        while(is_bank_confl && first != NULL)
        {
            int target_bank_id = first->bank_id;
            if ((controller->bank_status)[target_bank_id].next_free <= controller->cur_clk)
            {
                first->begin_exe = controller->cur_clk;
                controller->access_time += controller->cur_clk - first->queued_time;
                if (first->req_type == READ) {
                    first->end_exe = first->begin_exe + (uint64_t)controller->nclks_read;
                }
                else if (first->req_type == WRITE) {
                    first->end_exe = first->begin_exe + (uint64_t)controller->nclks_write;
                }
                // The target bank is no longer free until this request completes.
                (controller->bank_status)[target_bank_id].next_free = first->end_exe;

                migrateToQueue(controller->pending_queue, first);
                deleteNode(controller->waiting_queue, first);
                is_bank_confl = 0;
            } else {
                if (*conflict_req != new_first->queued_time && !is_incr_bank_confl) {
                    ++controller->bank_conficts;
                    is_incr_bank_confl = 1;
                }
                if (controller->is_frfcfs) {
                    first = first->next;
                }
            }
            if (!controller->is_frfcfs) {
                break;
            }
        }
        // printf("%ld\n", *conflict_req); fflush(stdout);
        *conflict_req = new_first->queued_time;
        
    }
}

#endif
