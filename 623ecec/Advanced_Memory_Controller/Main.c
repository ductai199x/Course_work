#include "Trace.h"
#include "Mem_System.h"
#include "hash_table.h"

#include<string.h>

extern TraceParser *initTraceParser(const char *mem_file);
extern bool getRequest(TraceParser *mem_trace);

extern MemorySystem *initMemorySystem();
extern unsigned pendingRequests(MemorySystem *mem_system);
extern bool access(MemorySystem *mem_system, Request *req);
extern void tickEvent(MemorySystem *mem_system);

typedef struct Mem_Timing
{
    unsigned nclks_read;
    unsigned nclks_write;
    unsigned nclks_channel;
    // unsigned nclks_rowbuff;

} Mem_Timing;

typedef struct Controller_Configs
{
    unsigned row;
    Mem_Timing dram_timing;
    unsigned max_waiting_queue_size; // Size of a cache line (in Bytes)
    unsigned block_size;             // Size of a cache (in KB)
    unsigned *num_of_banks;
    unsigned num_of_channels;
    bool is_bliss;
    bool is_ARSR;
    int ARSR_core_id;
    bool is_shared;

} Controller_Configs;

int main(int argc, const char *argv[])
{
    if (argc != 2)
    {
        printf("Usage: %s %s\n", argv[0], "<mem-file>");

        return 0;
    }

    Controller_Config config;

    Controller_Configs *bliss_dram_shared = malloc(sizeof(Controller_Configs));
    bliss_dram_shared->dram_timing.nclks_read = 53;
    bliss_dram_shared->dram_timing.nclks_write = 53;
    bliss_dram_shared->dram_timing.nclks_channel = 15;
    // bliss_dram->dram_timing.nclks_rowbuff = 53;
    unsigned bliss_dram_num_of_banks[] = {8, 16};
    bliss_dram_shared->max_waiting_queue_size = 64;
    bliss_dram_shared->block_size = 128;
    bliss_dram_shared->num_of_banks = bliss_dram_num_of_banks;
    bliss_dram_shared->num_of_channels = 4;
    bliss_dram_shared->is_bliss = true;
    bliss_dram_shared->is_ARSR = true;
    // bliss_dram_shared->ARSR_core_id = 2;
    bliss_dram_shared->is_shared = true;
    bliss_dram_shared->row = sizeof(bliss_dram_num_of_banks) / sizeof(bliss_dram_num_of_banks[0]);

    Controller_Configs *frfcfs_dram_shared = malloc(sizeof(Controller_Configs));
    memcpy(frfcfs_dram_shared, bliss_dram_shared, sizeof(Controller_Configs));
    frfcfs_dram_shared->is_bliss = false;

    Controller_Configs *controller_configs[] = { frfcfs_dram_shared, bliss_dram_shared };
    // Controller_Configs *controller_configs[] = { frfcfs_dram_shared };

    int num_tables = (int)(sizeof(controller_configs) / sizeof(controller_configs[0]));
    printf("bliss?, slwdwn, nbanks, nchans, clk_ch,  clk_r,  clk_w, unfairness\n");
    for (int t = 0; t < num_tables; t++)
    {
        for (int r = 0; r < controller_configs[t]->row; r++)
        {
            config.is_bliss = controller_configs[t]->is_bliss;
            config.is_ARSR = controller_configs[t]->is_ARSR;
            config.ARSR_core_id = controller_configs[t]->ARSR_core_id;
            config.is_shared = controller_configs[t]->is_shared;
            config.max_waiting_queue_size = controller_configs[t]->max_waiting_queue_size;
            config.block_size = controller_configs[t]->block_size;
            config.num_of_banks = controller_configs[t]->num_of_banks[r];
            config.num_of_channels = controller_configs[t]->num_of_channels;
            config.nclks_read = controller_configs[t]->dram_timing.nclks_read;
            config.nclks_write = controller_configs[t]->dram_timing.nclks_write;
            config.nclks_channel = controller_configs[t]->dram_timing.nclks_channel;
            // config.nclks_rowbuff = controller_configs[t]->dram_timing.nclks_rowbuff;
            printf("%6u, %6u, %6d, %6u, %6u, %6u, %6u, ", config.is_bliss, config.is_ARSR,
                config.num_of_channels, config.num_of_banks, config.nclks_channel, 
                config.nclks_read, config.nclks_write); fflush(stdout);

            int core_id = 0;
            int max_ncore = 8;
            uint64_t nreqs_shared = 0;

            hash_table *stalls_table = (hash_table* )malloc(sizeof(hash_table));
            double* slowdown_table = (double* )malloc(sizeof(double)*max_ncore);
            
            do {
                config.ARSR_core_id = core_id;
                // Initialize a CPU trace parser
                TraceParser *mem_trace = initTraceParser(argv[1]);

                // Initialize the memory system
                MemorySystem *mem_system = initMemorySystem(&config);

                uint64_t cycles = 0;
                uint64_t n_req = 0;

                bool stall = false;
                bool end = false;
                while (!end || pendingRequests(mem_system))
                {
                    if (!end && !stall)
                    {
                        end = !(getRequest(mem_trace));
                        while (!end && !config.is_shared && mem_trace->cur_req->core_id != core_id) {
                            end = !(getRequest(mem_trace));
                        }
                        n_req = end ? n_req : ++n_req;
                    }
                    if (!end)
                    {
                        stall = !(access(mem_system, mem_trace->cur_req));
                    }
                    tickEvent(mem_system);
                    ++cycles;
                        
                }

                if (config.is_shared) {
					nreqs_shared = n_req;	
                    memcpy(stalls_table, mem_system->stalls_table, sizeof(hash_table));
                } else {
                    slowdown_table[core_id] = ((double)hash_table_lookup(stalls_table, core_id)->stall_cycles/nreqs_shared) / (hash_table_lookup(mem_system->stalls_table, core_id)->stall_cycles/n_req);
                    core_id++;
                }

                
                config.is_shared = false;
		        config.is_bliss = false;
                free(mem_system->stalls_table);
                

                for (int i = 0; i < config.num_of_channels; i++) {
                    free(mem_system->controllers[i]);
                }

                free(mem_system);
                
            } while(config.is_ARSR && core_id < max_ncore);

            for (int i = 0; i < max_ncore; i++) {
                printf("%5lf, ", slowdown_table[i]);
            }

            printf("\n"); fflush(stdout);
            free(slowdown_table);
            free(stalls_table);
        }
    }
}
