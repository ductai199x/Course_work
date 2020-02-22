#ifndef __CONTROLLER_HH__
#define __CONTROLLER_HH__

#include "Bank.h"
#include "Queue.h"
#include "BLISS_table.h"

// Bank
extern void initBank(Bank *bank);

// Queue operations
extern Queue *initQueue();
extern void pushToQueue(Queue *q, Request *req);
extern void migrateToQueue(Queue *q, Node *_node);
extern void deleteNode(Queue *q, Node *node);

// Controller configs
typedef struct Controller_Config
{
    unsigned nclks_channel;
    unsigned nclks_read;
    unsigned nclks_write;
    // unsigned nclks_rowbuff;
    unsigned max_waiting_queue_size; // Size of a cache line (in Bytes)
    unsigned block_size;             // Size of a cache (in KB)
    unsigned num_of_banks;
    unsigned num_of_channels;
    bool is_bliss;
    bool is_ARSR;
    int ARSR_core_id;
    bool is_shared;
} Controller_Config;

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

    BLISS_table *bliss_table;

    /* For decoding */
    unsigned bank_shift;
    uint64_t bank_mask;

    unsigned block_size; // Size of a cache (in KB)
    unsigned nclks_channel;
    unsigned nclks_read;
    unsigned nclks_write;
    // unsigned nclks_rowbuff;
    unsigned max_waiting_queue_size; // Size of a cache line (in Bytes)
    unsigned num_of_banks;
    unsigned num_of_channels;

    bool is_shared;
    uint64_t IPC; 
    // Number of useful cycles (cycle where a mem req is served)
    // in a program. This is use to calculate Weighted Speedup
    
    bool is_ARSR;
    int ARSR_core_id;
    uint64_t ARSR_req_HP; // Request with Highest Priority
    uint64_t ARSR_cycle_HP; // Cycles with Highest Priority

    unsigned ctrl_id;
    bool is_bliss;

} Controller;

Controller *initController(Controller_Config *config)
{
    Controller *controller = (Controller *)malloc(sizeof(Controller));

    controller->block_size = config->block_size;
    controller->nclks_read = config->nclks_read;
    controller->nclks_write = config->nclks_write;
    controller->nclks_channel = config->nclks_channel;
    // controller->nclks_rowbuff = config->nclks_rowbuff;
    controller->max_waiting_queue_size = config->max_waiting_queue_size; // Size of a cache line (in Bytes)
    controller->is_bliss = config->is_bliss;
    controller->is_ARSR = config->is_ARSR;
    controller->ARSR_core_id = config->ARSR_core_id;
    controller->is_shared = config->is_shared;
    controller->num_of_banks = config->num_of_banks;
    controller->num_of_channels = config->num_of_channels;

    controller->ARSR_req_HP = 0;
    controller->ARSR_cycle_HP = 0;

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
    req->queued_time = controller->cur_clk;

    // Push to queue
    pushToQueue(controller->waiting_queue, req);

    return true;
}

int executeRequest(Controller *controller, Node *node)
{
    int target_bank_id = node->bank_id;
    if ((controller->bank_status)[target_bank_id].next_free <= controller->cur_clk &&
        controller->channel_next_free <= controller->cur_clk)
    {
        node->begin_exe = controller->cur_clk;
        // if (node->mem_addr == controller->bliss_table->addr_just_served)
        // {
        //     node->end_exe = node->begin_exe + (uint64_t)controller->nclks_rowbuff;
        // }
        // else
        // {
            if (node->req_type == READ)
            {
                node->end_exe = node->begin_exe + (uint64_t)controller->nclks_read;
            }
            else if (node->req_type == WRITE)
            {
                node->end_exe = node->begin_exe + (uint64_t)controller->nclks_write;
            }
        // }

        // The target bank is no longer free until this request completes.
        (controller->bank_status)[target_bank_id].next_free = node->end_exe;

        controller->channel_next_free = controller->cur_clk + controller->nclks_channel;

        if (node->core_id == controller->bliss_table->core_id) {
            if (++controller->bliss_table->num_req_served >= controller->bliss_table->blacklist_threshold) {
                controller->bliss_table->num_req_served = 0;
                controller->bliss_table->blacklisted_ids[node->core_id] = 1;
            }
        } else {
            controller->bliss_table->core_id = node->core_id;
            controller->bliss_table->num_req_served = 0;
        }
        controller->bliss_table->addr_just_served = node->mem_addr;

        migrateToQueue(controller->pending_queue, node);
        deleteNode(controller->waiting_queue, node);
        return 0;
    }

    return 1;
}

Node* better_request_BLISS(Controller* controller, Node* req1, Node* req2)
{
    if ((controller->bliss_table->blacklisted_ids[req1->core_id] == 0) ^ 
        (controller->bliss_table->blacklisted_ids[req2->core_id] == 0)) 
    {
        if ((controller->bliss_table->blacklisted_ids[req1->core_id] =! 1))
            return req1;
        else
            return req2;
    }
    
    bool is_req1_freehit = ((controller->bank_status)[req1->bank_id].next_free <= controller->cur_clk) &
        (controller->channel_next_free <= controller->cur_clk);
    bool is_req2_freehit = ((controller->bank_status)[req2->bank_id].next_free <= controller->cur_clk) &
        (controller->channel_next_free <= controller->cur_clk);

    if (is_req1_freehit ^ is_req2_freehit) {
        if (is_req1_freehit)
            return req1;
        else
            return req2;
    }

    if (req1->queued_time <= req2->queued_time) {
        return req1;
    } else {
        return req2;
    }
}

Node* better_request_FRFCFS(Controller* controller, Node* req1, Node* req2)
{
    bool is_req1_freehit = ((controller->bank_status)[req1->bank_id].next_free <= controller->cur_clk) &
        (controller->channel_next_free <= controller->cur_clk);
    bool is_req2_freehit = ((controller->bank_status)[req2->bank_id].next_free <= controller->cur_clk) &
        (controller->channel_next_free <= controller->cur_clk);

    if (is_req1_freehit ^ is_req2_freehit) {
        if (is_req1_freehit)
            return req1;
        else
            return req2;
    }

    if (req1->queued_time <= req2->queued_time) {
        return req1;
    } else {
        return req2;
    }
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

    int is_fail_executed = 1;
    // Step three, find a request to schedule
    if (controller->waiting_queue->size)
    {
        Node *first = controller->waiting_queue->first;
        Node *best_node = first;
        Node *next_node = first->next;
        // BLISS
        if (controller->is_bliss)
        {
            while (next_node != NULL)
            {
                best_node = better_request_BLISS(controller, best_node, next_node);
                next_node = next_node->next;
            }
            
            if (controller->cur_clk % controller->bliss_table->clearing_interval == 0)
            {
                for (int i = 0; i < 10; i++)
                    controller->bliss_table->blacklisted_ids[i] = 0;
            }
        }
        // FRFCFS
        else
        {
            while (next_node != NULL)
            {
                best_node = better_request_FRFCFS(controller, best_node, next_node);
                next_node = next_node->next;
            }
        }

        bool ARSR = (controller->is_ARSR) & (best_node->core_id == controller->ARSR_core_id);

        is_fail_executed = executeRequest(controller, best_node);

        if (ARSR) {
            ++controller->ARSR_req_HP;
            if (!is_fail_executed) {
                ++controller->ARSR_cycle_HP;
            }
        }

        if (is_fail_executed) {

        }
        
    }
}

#endif
