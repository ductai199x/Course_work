#ifndef __MEMORY_SYSTEM_HH__
#define __MEMORY_SYSTEM_HH__

#include "Controller.h"

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
}MemorySystem;

MemorySystem *initMemorySystem(Controller_Config* config)
{
    MemorySystem *mem_system = (MemorySystem *)malloc(sizeof(MemorySystem));

    mem_system->controllers = (Controller **)malloc(config->num_of_channels * sizeof(Controller *));
    int i;
    for (i = 0; i < config->num_of_channels; i++)
    {
        mem_system->controllers[i] = initController(config);
    }

    mem_system->channel_shift = log2(config->block_size);
    mem_system->channel_mask = (uint64_t)config->num_of_channels - (uint64_t)1;

    return mem_system;
}

unsigned pendingRequests(MemorySystem *mem_system)
{
    unsigned num_reqs_left = 0;
    int i;
    for (i = 0; i < mem_system->controllers[i]->num_of_channels; i++)
    {
        num_reqs_left += ongoingPendingRequests(mem_system->controllers[i]);
    }

    return num_reqs_left;
}

bool access(MemorySystem *mem_system, Request *req)
{
    unsigned channel_id = ((req->memory_address) >> mem_system->channel_shift) 
        & mem_system->channel_mask;

    req->channel_id = channel_id;
    return send(mem_system->controllers[channel_id], req);
}

void tickEvent(MemorySystem *mem_system)
{
    int i;
    for (i = 0; i < mem_system->controllers[i]->num_of_channels; i++)
    {
        tick(mem_system->controllers[i]);
    }
}

#endif
