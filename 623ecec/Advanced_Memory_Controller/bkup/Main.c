#include "Trace.h"
#include "Mem_System.h"

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
    unsigned bliss_dram_num_of_banks[] = {16};
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

    Controller_Configs *controller_configs[] = { bliss_dram_shared, frfcfs_dram_shared};

    int num_tables = (int)(sizeof(controller_configs) / sizeof(controller_configs[0]));
    // printf("max_queue, blk_size, nbanks, nchannels, clk_r, clk_w, clk_ch, nreqs, avg_at, bank_cfl, exec_time\n");
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
            printf("%u, %u, %d %u, %u, %u, %u, %u, %u, ", config.is_bliss, config.is_ARSR, config.ARSR_core_id, 
                config.num_of_channels, config.is_shared, config.num_of_banks, config.nclks_channel, 
                config.nclks_read, config.nclks_write);

            double maximum_slowdown = 0;
            int core_id = 0;
            do {
                double ARSR = 0;
                double SRSR = 0;
                double alpha = 0;
                double slowdown = 0;
                config.ARSR_core_id = core_id;
                // Initialize a CPU trace parser
                TraceParser *mem_trace = initTraceParser(argv[1]);

                // Initialize the memory system
                MemorySystem *mem_system = initMemorySystem(&config);

                uint64_t cycles = 0;
                uint64_t n_req = 0;
                uint64_t stall_cycles = 0;

                bool stall = false;
                bool end = false;
                while (!end || pendingRequests(mem_system))
                {
                    if (!end && !stall)
                    {
                        end = !(getRequest(mem_trace));
                        n_req = end ? n_req : ++n_req;
                    }
                    if (!end)
                    {
                        stall = !(access(mem_system, mem_trace->cur_req));
                        stall_cycles = stall ? ++stall_cycles : stall_cycles;
                    }
                    tickEvent(mem_system);
                    ++cycles;
                }
                
                for (int i = 0; i < config.num_of_channels; i++) {
                    ARSR += (double)mem_system->controllers[i]->ARSR_cycle_HP/mem_system->controllers[i]->ARSR_req_HP;
                }
                SRSR = (double)n_req/cycles;
                alpha = (double)stall_cycles/cycles;
                // slowdown = (1 - alpha) + alpha*ARSR/SRSR;
                slowdown = ARSR/SRSR;
                printf("coreid: %d, ARSR: %f, SRSR: %f, alpha: %f, slowdown: %f\n", core_id, ARSR, SRSR, alpha, slowdown);

                maximum_slowdown = slowdown > maximum_slowdown ? slowdown : maximum_slowdown;
                core_id++;

                free(mem_system);
            } while(config.is_ARSR && core_id < 8);
            
        }
    }
}
