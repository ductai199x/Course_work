#include "Trace.h"
#include "Cache.h"

extern TraceParser *initTraceParser(const char * mem_file);
extern bool getRequest(TraceParser *mem_trace);

extern Cache *initCache(Cache_Config* config);
extern bool accessBlock(Cache *cache, Request *req, uint64_t access_time);
extern bool insertBlock(Cache *cache, Request *req, Cache_Config *config, uint64_t access_time, uint64_t *wb_addr);

typedef struct Cache_Table
{
    int row;
    unsigned* block_size; // Size of a cache line (in Bytes)
    unsigned* cache_size; // Size of a cache (in KB)
    unsigned* assoc;

    char* name;
}Cache_Table;

int main(int argc, const char *argv[])
{	
    if (argc != 2)
    {
        printf("Usage: %s %s\n", argv[0], "<mem-file>");

        return 0;
    }

    Cache_Config config = (Cache_Config) {
        .block_size=64,
        .cache_size=512*1024,
        .assoc=4
    };

    Cache_Table *LRU_table = malloc(sizeof(Cache_Table));
    LRU_table->name = "lru";
    unsigned block_size_lru[] = {64, 64, 64, 64, 64};
    unsigned cache_size_lru[] = {128*1024, 256*1024, 512*1024, 1024*1024, 2048*1024};
    unsigned assoc_lru[] = {4, 4, 4, 4, 4};
    LRU_table->row = sizeof(block_size_lru)/sizeof(block_size_lru[0]);
    LRU_table->block_size = block_size_lru;
    LRU_table->cache_size = cache_size_lru;
    LRU_table->assoc = assoc_lru;

    Cache_Table *LFU_table = malloc(sizeof(Cache_Table));
    LFU_table->name = "lfu";
    unsigned block_size_lfu[] = {64, 64, 64, 64, 64};
    unsigned cache_size_lfu[] = {128*1024, 256*1024, 512*1024, 1024*1024, 2048*1024};
    unsigned assoc_lfu[] = {4, 4, 4, 4, 4};
    LFU_table->row = sizeof(block_size_lfu)/sizeof(block_size_lfu[0]);
    LFU_table->block_size = block_size_lfu;
    LFU_table->cache_size = cache_size_lfu;
    LFU_table->assoc = assoc_lfu;

    Cache_Table *cache_tables[] = { LRU_table, LFU_table };

    int num_tables = (int)( sizeof(cache_tables) / sizeof(cache_tables[0]) );

    for (int t = 0; t < num_tables; t++ ) {
        config.replacement_policy = cache_tables[t]->name;
        for (int r = 0; r < cache_tables[t]->row; r++) {
            if (!strcmp(cache_tables[t]->name, "lru")) {
                config.block_size = cache_tables[t]->block_size[r];
                config.cache_size = cache_tables[t]->cache_size[r];
                config.assoc = cache_tables[t]->assoc[r];
                printf("%s, %u, %u, %u, ", config.replacement_policy, config.block_size, config.cache_size/(1024), config.assoc);
            }
            else if (!strcmp(cache_tables[t]->name, "lfu")) {
                config.block_size = cache_tables[t]->block_size[r];
                config.cache_size = cache_tables[t]->cache_size[r];
                config.assoc = cache_tables[t]->assoc[r];
                printf("%s, %u, %u, %u, ", config.replacement_policy, config.block_size, config.cache_size/(1024), config.assoc);
            }
            else {
                printf("Table not implemented!");
                return 1;
            }

            // Initialize a CPU trace parser
            TraceParser *mem_trace = initTraceParser(argv[1]);

            // Initialize a Cache
            Cache *cache = initCache(&config);

            // Running the trace
            uint64_t num_of_reqs = 0;
            uint64_t hits = 0;
            uint64_t misses = 0;
            uint64_t num_evicts = 0;

            uint64_t cycles = 0;
            while (getRequest(mem_trace))
            {
                // Step one, accessBlock()
                if (accessBlock(cache, mem_trace->cur_req, cycles))
                {
                    // Cache hit
                    hits++;
                }
                else
                {
                    // Cache miss!
                    misses++;
                    // Step two, insertBlock()
        //            printf("Inserting: %"PRIu64"\n", mem_trace->cur_req->load_or_store_addr);
                    uint64_t wb_addr;
                    if (insertBlock(cache, mem_trace->cur_req, &config, cycles, &wb_addr))
                    {
                        num_evicts++;
        //                printf("Evicted: %"PRIu64"\n", wb_addr);
                    }
                }

                ++num_of_reqs;
                ++cycles;
            }

            double hit_rate = (double)hits / ((double)hits + (double)misses);
            printf("Hit rate: %lf%%\n", hit_rate * 100);
        }
    }
    
}
