#ifndef __MEMORY_SYSTEM_HH__
#define __MEMORY_SYSTEM_HH__

#include "Controller.h"
#include "BLISS_table.h"

extern Controller *initController();
extern unsigned ongoingPendingRequests(Controller *controller);
extern bool send(Controller *controller, Request *req);
extern void tick(Controller *controller);

typedef struct MemorySystem
{
    Controller **controllers; // All the channels/controllers in the memory system

    /* For decoding */
    unsigned channel_shift;
    uint64_t channel_mask;

    unsigned num_of_channels;
    BLISS_table *bliss_table;

} MemorySystem;

MemorySystem *initMemorySystem(Controller_Config *config)
{
    MemorySystem *mem_system = (MemorySystem *)malloc(sizeof(MemorySystem));

    mem_system->controllers = (Controller **)malloc(config->num_of_channels * sizeof(Controller *));
    mem_system->bliss_table = (BLISS_table* )malloc(sizeof(BLISS_table));
    mem_system->bliss_table->blacklist_threshold = 4;
    mem_system->bliss_table->clearing_interval = 10000;
    mem_system->bliss_table->blacklisted_ids = (int* )malloc(sizeof(int)*10);

    for (int i = 0; i < config->num_of_channels; i++)
    {
        mem_system->controllers[i] = initController(config);
        mem_system->controllers[i]->ctrl_id = i;
        mem_system->controllers[i]->bliss_table = mem_system->bliss_table;
    }
    mem_system->num_of_channels = config->num_of_channels;
    mem_system->channel_shift = log2(config->block_size);
    mem_system->channel_mask = (uint64_t)config->num_of_channels - (uint64_t)1;

    return mem_system;
}

unsigned pendingRequests(MemorySystem *mem_system)
{
    unsigned num_reqs_left = 0;

    for (int i = 0; i < mem_system->num_of_channels; i++)
    {
        num_reqs_left += ongoingPendingRequests(mem_system->controllers[i]);
    }

    return num_reqs_left;
}

bool access(MemorySystem *mem_system, Request *req)
{
    unsigned channel_id = ((req->memory_address) >> mem_system->channel_shift) & mem_system->channel_mask;

    req->channel_id = channel_id;
    return send(mem_system->controllers[channel_id], req);
}

void tickEvent(MemorySystem *mem_system)
{

    for (int i = 0; i < mem_system->num_of_channels; i++)
    {
        tick(mem_system->controllers[i]);
    }
}

#endif
