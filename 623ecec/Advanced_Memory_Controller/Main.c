#include "Trace.h"

#include "Mem_System.h"

extern TraceParser *initTraceParser(const char * mem_file);
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

}Mem_Timing;

typedef struct Controller_Configs
{
    unsigned row;
    Mem_Timing dram_timing;
    unsigned* max_waiting_queue_size; // Size of a cache line (in Bytes)
    unsigned* block_size; // Size of a cache (in KB)
    unsigned* num_of_banks;
    unsigned num_of_channels;
    // bool is_frfcfs;
    
}Controller_Configs;

int main(int argc, const char *argv[])
{	
    if (argc != 2)
    {
        printf("Usage: %s %s\n", argv[0], "<mem-file>");

        return 0;
    }

    Controller_Config config;

    Controller_Configs *bliss_dram = malloc(sizeof(Controller_Configs));
    bliss_dram->dram_timing.nclks_read = 53;
    bliss_dram->dram_timing.nclks_write = 53;
    bliss_dram->dram_timing.nclks_channel = 15;
    unsigned bliss_dram_max_waiting_queue_size[] = {64, 64, 64, 64};
    unsigned bliss_dram_block_size[] = {128, 128, 128, 128};
    unsigned bliss_dram_num_of_banks[] = {2, 4, 8, 16};
    bliss_dram->max_waiting_queue_size = bliss_dram_max_waiting_queue_size;
    bliss_dram->block_size = bliss_dram_block_size;
    bliss_dram->num_of_banks = bliss_dram_num_of_banks;
    bliss_dram->num_of_channels = 4;
    // fcfs_dram->is_frfcfs = false;
    bliss_dram->row = sizeof(bliss_dram_num_of_banks)/sizeof(bliss_dram_num_of_banks[0]);


    Controller_Configs *controller_configs[] = { bliss_dram };
    // Controller_Table *controller_tables[] = { fcfs_dram, fcfs_pcm, frfcfs_dram, frfcfs_pcm };
    int num_tables = (int)( sizeof(controller_configs) / sizeof(controller_configs[0]) );
    printf("max_queue, blk_size, nbanks, nchannels, clk_r, clk_w, clk_ch, nreqs, avg_at, bank_cfl, exec_time\n");
        for (int t = 0; t < num_tables; t++ ) {
            for (int r = 0; r < controller_configs[t]->row; r++) {
                // config.is_frfcfs = controller_configs[t]->is_frfcfs;
                config.max_waiting_queue_size = controller_configs[t]->max_waiting_queue_size[r];
                config.block_size = controller_configs[t]->block_size[r];
                config.num_of_banks = controller_configs[t]->num_of_banks[r];
                config.num_of_channels = controller_configs[t]->num_of_channels;
                config.nclks_read = controller_configs[t]->dram_timing.nclks_read;
                config.nclks_write = controller_configs[t]->dram_timing.nclks_write;
                config.nclks_channel = controller_configs[t]->dram_timing.nclks_channel;
                printf("%6u, %9u, %8u, %6u, %5u, %5u, %5u, ", config.max_waiting_queue_size, 
                    config.block_size, config.num_of_channels, config.num_of_banks, config.nclks_channel, config.nclks_read, config.nclks_write);

                // Initialize a CPU trace parser
                TraceParser *mem_trace = initTraceParser(argv[1]);

                // Initialize the memory system
                MemorySystem *mem_system = initMemorySystem(&config);
                // printf("%u\n", controller->bank_shift);
                // printf("%u\n", controller->bank_mask);

                uint64_t cycles = 0;

                bool stall = false;   
                bool end = false;

                while (!end || pendingRequests(mem_system))
                {
                    
                    if (!end && !stall)
                    {
                        end = !(getRequest(mem_trace));
                    }

                    if (!end)
                    {
                        stall = !(access(mem_system, mem_trace->cur_req));
                    
                        // printf("%u ", mem_trace->cur_req->core_id);
                        // printf("%u ", mem_trace->cur_req->req_type);
                        // printf("%"PRIu64" \n", mem_trace->cur_req->memory_address);
                    }
// printf("lmaoo"); fflush(stdout);
                    tickEvent(mem_system);
                    ++cycles;
                    
                }

                // TODO, de-allocate memory
                /*
                free(controller->bank_status);
                free(controller->waiting_queue);
                free(controller->pending_queue);
                free(controller);
                */
                printf("End Execution Time: ""%"PRIu64"\n", cycles);
            }
        }
}
