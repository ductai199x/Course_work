#ifndef __CACHE_H__
#define __CACHE_H__

#include <assert.h>

#include <stdio.h>
#include <stdlib.h>

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "Cache_Blk.h"
#include "Request.h"

#define LRU

/* Cache */
typedef struct Set
{
    Cache_Block **ways; // Block ways within a set
}Set;

typedef struct Sat_Counter
{
    uint8_t max_val;
    uint8_t counter;
}Sat_Counter;

typedef struct Cache_Config
{
    unsigned block_size; // Size of a cache line (in Bytes)
    unsigned cache_size; // Size of a cache (in KB)
    unsigned assoc;

    char* replacement_policy;
}Cache_Config;


typedef struct Cache
{
    uint64_t blk_mask;
    unsigned num_blocks;
    
    Cache_Block *blocks; // All cache blocks

    Sat_Counter *SHCT;  //  Signature Hit Counter Table
    uint64_t SHCT_mask;

    /* Set-Associative Information */
    unsigned num_sets; // Number of sets
    unsigned num_ways; // Number of ways within a set

    unsigned set_shift;
    unsigned set_mask; // To extract set index
    unsigned tag_shift; // To extract tag

    Set *sets; // All the sets of a cache
    
}Cache;

// Function Definitions
Cache *initCache(Cache_Config* config);
bool accessBlock(Cache *cache, Request *req, uint64_t access_time);
bool insertBlock(Cache *cache, Request *req, Cache_Config *config, uint64_t access_time, uint64_t *wb_addr);

// Helper Function
uint64_t blkAlign(uint64_t addr, uint64_t mask);
uint64_t calculatePCSignature(Cache_Block *blk, uint64_t mask);
Cache_Block *findBlock(Cache *cache, uint64_t addr);
void incrementSHCT(Cache_Block *blk, Cache *cache);
void decrementSHCT(Cache_Block *blk, Cache *cache);

// Replacement Policies
bool lru(Cache *cache, uint64_t addr, Cache_Block **victim_blk, uint64_t *wb_addr);
bool lfu(Cache *cache, uint64_t addr, Cache_Block **victim_blk, uint64_t *wb_addr);

#endif
