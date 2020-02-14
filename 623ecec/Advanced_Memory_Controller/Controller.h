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

// CONSTANTS
static unsigned MAX_WAITING_QUEUE_SIZE = 64;
static unsigned BLOCK_SIZE = 64; // cache block size
static unsigned NUM_OF_CHANNELS = 4; // 4 channels/controllers in total
static unsigned NUM_OF_BANKS = 32; // number of banks per channel

// DRAM Timings
static unsigned nclks_channel = 15;
static unsigned nclks_read = 53;
static unsigned nclks_write = 53;

// PCM Timings
// static unsigned nclks_read = 57;
// static unsigned nclks_write = 162;

// Controller configs
typedef struct Controller_Config
{
    unsigned nclks_channel;
    unsigned nclks_read;
    unsigned nclks_write;
    unsigned max_waiting_queue_size; // Size of a cache line (in Bytes)
    unsigned block_size; // Size of a cache (in KB)
    unsigned num_of_banks;
    unsigned num_of_channels;
    bool is_frfcfs;
}Controller_Config;


// Controller definition
typedef struct Controller
{
    // The memory controller needs to maintain records of all bank's status
    Bank *bank_status;

    // Current memory clock
    uint64_t cur_clk;

    // Channel status
    uint64_t channel_next_free;

    // A queue contains all the requests that are waiting to be issued.
    Queue *waiting_queue;

    // A queue contains all the requests that have already been issued 
    // but are waiting to complete.
    Queue *pending_queue;

    /* For decoding */
    unsigned bank_shift;
    uint64_t bank_mask;

    unsigned block_size; // Size of a cache (in KB)
    unsigned nclks_channel;
    unsigned nclks_read;
    unsigned nclks_write;
    unsigned max_waiting_queue_size; // Size of a cache line (in Bytes)
    unsigned num_of_banks;
    unsigned num_of_channels;

}Controller;

Controller *initController(Controller_Config* config)
{
    Controller *controller = (Controller *)malloc(sizeof(Controller));

    controller->block_size = config->block_size;
    controller->nclks_read = config->nclks_read;
    controller->nclks_write = config->nclks_write;
    controller->max_waiting_queue_size = config->max_waiting_queue_size; // Size of a cache line (in Bytes)
    controller->nclks_channel = config->nclks_channel;
    controller->num_of_banks = config->num_of_banks;
    controller->num_of_channels = config->num_of_channels;

    controller->bank_status = (Bank *)malloc(config->num_of_banks * sizeof(Bank));
    for (int i = 0; i < config->num_of_banks; i++)
    {
        initBank(&((controller->bank_status)[i]));
    }
    controller->cur_clk = 0;
    controller->channel_next_free = 0;

    controller->waiting_queue = initQueue();
    controller->pending_queue = initQueue();

    controller->bank_shift = log2(config->block_size) + log2(config->num_of_channels);
    controller->bank_mask = (uint64_t)config->num_of_banks - (uint64_t)1;

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
    
    // Push to queue
    pushToQueue(controller->waiting_queue, req);

    return true;
}

void tick(Controller *controller)
{
    // Step one, update system stats
    ++(controller->cur_clk);
    // printf("Clk: ""%"PRIu64"\n", controller->cur_clk);
    for (int i = 0; i < controller->num_of_banks; i++)
    {
        ++(controller->bank_status)[i].cur_clk;
        // printf("%"PRIu64"\n", (controller->bank_status)[i].cur_clk);
    }
    // printf("\n");

    // Step two, serve pending requests
    if (controller->pending_queue->size)
    {
        Node *first = controller->pending_queue->first;
        if (first->end_exe <= controller->cur_clk)
        {
            /*
            printf("Clk: ""%"PRIu64"\n", controller->cur_clk);
            printf("Address: ""%"PRIu64"\n", first->mem_addr);
            printf("Channel ID: %d\n", first->channel_id);
            printf("Bank ID: %d\n", first->bank_id);
            printf("Begin execution: ""%"PRIu64"\n", first->begin_exe);
            printf("End execution: ""%"PRIu64"\n\n", first->end_exe);
            */

            deleteNode(controller->pending_queue, first);
        }
    }

    // Step three, find a request to schedule
    if (controller->waiting_queue->size)
    {
        // Implementation One - FCFS
        Node *first = controller->waiting_queue->first;
        int target_bank_id = first->bank_id;

        if ((controller->bank_status)[target_bank_id].next_free <= controller->cur_clk && 
            controller->channel_next_free <= controller->cur_clk)
        {
            first->begin_exe = controller->cur_clk;
            if (first->req_type == READ)
            {
                first->end_exe = first->begin_exe + (uint64_t)controller->nclks_read;
            }
            else if (first->req_type == WRITE)
            {
                first->end_exe = first->begin_exe + (uint64_t)controller->nclks_write;
            }
            // The target bank is no longer free until this request completes.
            (controller->bank_status)[target_bank_id].next_free = first->end_exe;
            
            controller->channel_next_free = controller->cur_clk + controller->nclks_channel;

            migrateToQueue(controller->pending_queue, first);
            deleteNode(controller->waiting_queue, first);
        }
    }
}

#endif
