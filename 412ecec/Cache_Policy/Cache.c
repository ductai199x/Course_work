#include "Cache.h"

unsigned SHCT_size = 16*1024;

Cache *initCache(Cache_Config* config)
{
    Cache *cache = (Cache *)malloc(sizeof(Cache));

    cache->blk_mask = config->block_size - 1;

    unsigned num_blocks = config->cache_size * 1024 / config->block_size;
    cache->num_blocks = num_blocks;
    printf("%u, ", cache->num_blocks);

    // Initialize all cache blocks
    cache->blocks = (Cache_Block *)malloc(num_blocks * sizeof(Cache_Block));

    cache->SHCT = (Sat_Counter *)malloc(SHCT_size * sizeof(Sat_Counter));
    cache->SHCT_mask = SHCT_size-1;
    for (int i = 0; i < SHCT_size; i++)
    {
        cache->SHCT[i].counter = 0;
        cache->SHCT[i].max_val = 8;
    }
    

    for (int i = 0; i < num_blocks; i++)
    {
        cache->blocks[i].tag = UINTMAX_MAX; 
        cache->blocks[i].valid = false;
        cache->blocks[i].dirty = false;
        cache->blocks[i].when_touched = 0;
        cache->blocks[i].frequency = 0;
        cache->blocks[i].signature = 0;
        cache->blocks[i].outcome = false;
        cache->blocks[i].PC = 0;
        cache->blocks[i].load_store_addr = 0;
    }

    // Initialize Set-way variables
    unsigned num_sets = config->cache_size * 1024 / (config->block_size * config->assoc);
    cache->num_sets = num_sets;
    cache->num_ways = config->assoc;
    // printf("Num of sets: %u\n", cache->num_sets);

    unsigned set_shift = log2(config->block_size);
    cache->set_shift = set_shift;
    // printf("Set shift: %u\n", cache->set_shift);

    unsigned set_mask = num_sets - 1;
    cache->set_mask = set_mask;
    // printf("Set mask: %u\n", cache->set_mask);

    unsigned tag_shift = set_shift + log2(num_sets);
    cache->tag_shift = tag_shift;
    // printf("Tag shift: %u\n", cache->tag_shift);

    // Initialize Sets
    cache->sets = (Set *)malloc(num_sets * sizeof(Set));
    for (int i = 0; i < num_sets; i++)
    {
        cache->sets[i].ways = (Cache_Block **)malloc(config->assoc * sizeof(Cache_Block *));
    }

    // Combine sets and blocks
    for (int i = 0; i < num_blocks; i++)
    {
        Cache_Block *blk = &(cache->blocks[i]);
        
        uint32_t set = i / config->assoc;
        uint32_t way = i % config->assoc;

        blk->set = set;
        blk->way = way;

        cache->sets[set].ways[way] = blk;
    }

    return cache;
}

bool accessBlock(Cache *cache, Request *req, uint64_t access_time)
{
    bool hit = false;

    uint64_t blk_aligned_addr = blkAlign(req->load_or_store_addr, cache->blk_mask);

    Cache_Block *blk = findBlock(cache, blk_aligned_addr);
   
    if (blk != NULL) 
    {
        hit = true;

        // Update access time	
        blk->when_touched = access_time;
        // Increment frequency counter
        ++blk->frequency;

        blk->outcome = true;

        incrementSHCT(blk, cache);

        if (req->req_type == STORE)
        {
            blk->dirty = true;
        }
    }

    return hit;
}

bool insertBlock(Cache *cache, Request *req, Cache_Config *config, uint64_t access_time, uint64_t *wb_addr)
{
    // Step one, find a victim block
    uint64_t blk_aligned_addr = blkAlign(req->load_or_store_addr, cache->blk_mask);

    Cache_Block *victim = NULL;
    bool wb_required = false;
    if (!strcmp(config->replacement_policy, "lru")) {
        wb_required = lru(cache, blk_aligned_addr, &victim, wb_addr);
    } else if (!strcmp(config->replacement_policy, "lfu")) {
        wb_required = lfu(cache, blk_aligned_addr, &victim, wb_addr);
    } else {
        return 1;
    }

    assert(victim != NULL);

    uint64_t tag = req->load_or_store_addr >> cache->tag_shift;
    victim->tag = tag;
    // Step two, insert the new block
    victim->valid = true;
    victim->PC = req->PC;
    victim->load_store_addr = req->load_or_store_addr;
    victim->signature = calculatePCSignature(victim, cache->SHCT_mask);

    victim->when_touched = access_time;
    ++victim->frequency;

    if (req->req_type == STORE)
    {
        victim->dirty = true;
    }

    return wb_required;
}

bool lru(Cache *cache, uint64_t addr, Cache_Block **victim_blk, uint64_t *wb_addr)
{
    uint64_t set_idx = (addr >> cache->set_shift) & cache->set_mask;

    Cache_Block **ways = cache->sets[set_idx].ways;

    // Step one, try to find an invalid block.
    int i;
    for (i = 0; i < cache->num_ways; i++)
    {
        if (ways[i]->valid == false)
        {
            *victim_blk = ways[i];
            return false; // No need to write-back
        }
    }

    // Step two, if there is no invalid block. Locate the LRU block
    Cache_Block *victim = ways[0];
    uint64_t victim_shct = cache->SHCT[victim->signature].counter;
    for (i = 1; i < cache->num_ways; i++)
    {
        if (cache->SHCT[ways[i]->signature].counter < victim_shct) {
            victim = ways[i];
            victim_shct = cache->SHCT[ways[i]->signature].counter;
            continue;
        }
        if (cache->SHCT[ways[i]->signature].counter == victim_shct) {
            if (ways[i]->when_touched < victim->when_touched) {
                victim = ways[i];
            }
        }
    }

    if (victim->outcome == false) {
        decrementSHCT(victim, cache);
    }

    // Step three, need to write-back the victim block
    *wb_addr = (victim->tag << cache->tag_shift) | (victim->set << cache->set_shift);

    // Step three, invalidate victim
    victim->tag = UINTMAX_MAX;
    victim->valid = false;
    victim->dirty = false;
    victim->frequency = 0;
    victim->when_touched = 0;

    victim->outcome = false;
    victim->signature = 0;
    victim->PC = 0;

    *victim_blk = victim;

    return true; // Need to write-back
}

// Helper Functions
inline uint64_t blkAlign(uint64_t addr, uint64_t mask)
{
    return addr & ~mask;
}

uint64_t calculatePCSignature(Cache_Block *blk, uint64_t mask)
{
    // MASK & ((PC << 5) + ((accessType == ACCESS_PREFETCH) << 4) + ((accessType == ACCESS_WRITEBACK) << 3) + ((accessType == ACCESS_IFETCH) << 2) + ((accessType == ACCESS_LOAD) << 1)  + ((accessType == ACCESS_STORE) << 0)); 
    
    return mask & (blk->PC ^ blk->tag);
    // return mask & blk->PC;
}

Cache_Block *findBlock(Cache *cache, uint64_t addr)
{

    // Extract tag
    uint64_t tag = addr >> cache->tag_shift;

    // Extract set index
    uint64_t set_idx = (addr >> cache->set_shift) & cache->set_mask;

    Cache_Block **ways = cache->sets[set_idx].ways;
    int i;
    for (i = 0; i < cache->num_ways; i++)
    {
        if (tag == ways[i]->tag && ways[i]->valid == true)
        {
            return ways[i];
        }
    }

    return NULL;
}

void incrementSHCT(Cache_Block *blk, Cache *cache)
{
    uint64_t signature = calculatePCSignature(blk, cache->SHCT_mask);
    if (cache->SHCT[signature].counter < cache->SHCT[signature].max_val) {
        cache->SHCT[signature].counter++;
    }
}

void decrementSHCT(Cache_Block *blk, Cache *cache)
{
    uint64_t signature = calculatePCSignature(blk, cache->SHCT_mask);
    if (cache->SHCT[signature].counter > 0) {
        cache->SHCT[signature].counter--;
    }
}


bool lfu(Cache *cache, uint64_t addr, Cache_Block **victim_blk, uint64_t *wb_addr)
{
    uint64_t set_idx = (addr >> cache->set_shift) & cache->set_mask;
    //    printf("Set: %"PRIu64"\n", set_idx);
    Cache_Block **ways = cache->sets[set_idx].ways;

    // Step one, try to find an invalid block.
    int i;
    for (i = 0; i < cache->num_ways; i++)
    {
        if (ways[i]->valid == false)
        {
            *victim_blk = ways[i];
            return false; // No need to write-back
        }
    }

    // Step two, if there is no invalid block. Locate the LFU block
    Cache_Block *victim = ways[0];
    uint64_t victim_shct = cache->SHCT[victim->signature].counter;
    for (i = 1; i < cache->num_ways; i++)
    {
        if (cache->SHCT[ways[i]->signature].counter < victim_shct) {
            victim = ways[i];
            victim_shct = cache->SHCT[ways[i]->signature].counter;
            continue;
        }
        if (cache->SHCT[ways[i]->signature].counter == victim_shct) {
            if (ways[i]->frequency < victim->frequency) {
                victim = ways[i];
            }
        }
    }

    if (victim->outcome == false) {
        decrementSHCT(victim, cache);
    }

    // Step three, need to write-back the victim block
    *wb_addr = (victim->tag << cache->tag_shift) | (victim->set << cache->set_shift);

    // Step three, invalidate victim
    victim->tag = UINTMAX_MAX;
    victim->valid = false;
    victim->dirty = false;
    victim->frequency = 0;
    victim->when_touched = 0;

    victim->outcome = false;
    victim->signature = 0;
    victim->PC = 0;

    *victim_blk = victim;

    return true; // Need to write-back
}