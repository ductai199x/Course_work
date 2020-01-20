#include "Trace.h"
#include "Request.h"
#include "Controller.h"

extern TraceParser *initTraceParser(const char * mem_file);
extern bool getRequest(TraceParser *mem_trace);

extern Controller *initController();
extern unsigned ongoingPendingRequests(Controller *controller);
extern bool send(Controller *controller, Request *req);
extern void tick(Controller *controller, uint64_t* conflict_req);

typedef struct DRAM_Timing
{
    unsigned nclks_read;
    unsigned nclks_write;
}DRAM_Timing;

typedef struct Controller_Table
{
    unsigned row;
    DRAM_Timing dram_timing;
    unsigned* max_waiting_queue_size; // Size of a cache line (in Bytes)
    unsigned* block_size; // Size of a cache (in KB)
    unsigned* num_of_banks;
    bool is_fcfs;
    
    // char* name;
}Controller_Table;

int main(int argc, const char *argv[])
{	
    if (argc != 2)
    {
        printf("Usage: %s %s\n", argv[0], "<mem-file>");

        return 0;
    }

    Controller_Config config;

    Controller_Table *fcfs = malloc(sizeof(Controller_Table));
    fcfs->dram_timing.nclks_read = 53;
    fcfs->dram_timing.nclks_write = 53;
    unsigned max_waiting_queue_size[] = {64, 64, 64, 64};
    unsigned block_size[] = {128, 128, 128, 128};
    unsigned num_of_banks[] = {2, 4, 8, 16};
    fcfs->max_waiting_queue_size = max_waiting_queue_size;
    fcfs->block_size = block_size;
    fcfs->num_of_banks = num_of_banks;
    fcfs->is_fcfs = false;
    fcfs->row = sizeof(num_of_banks)/sizeof(num_of_banks[0]);

    Controller_Table *fcfs_pcm = malloc(sizeof(Controller_Table));
    fcfs_pcm->dram_timing.nclks_read = 57;
    fcfs_pcm->dram_timing.nclks_write = 162;
    unsigned max_waiting_queue_size_pcm[] = {64, 64, 64, 64};
    unsigned block_size_pcm[] = {128, 128, 128, 128};
    unsigned num_of_banks_pcm[] = {2, 4, 8, 16};
    fcfs_pcm->max_waiting_queue_size = max_waiting_queue_size_pcm;
    fcfs_pcm->block_size = block_size_pcm;
    fcfs_pcm->num_of_banks = num_of_banks_pcm;
    fcfs_pcm->is_fcfs = false;
    fcfs_pcm->row = sizeof(num_of_banks)/sizeof(num_of_banks[0]);

    Controller_Table *controller_tables[] = { fcfs, fcfs_pcm };
    int num_tables = (int)( sizeof(controller_tables) / sizeof(controller_tables[0]) );

    printf("fcfs, max_queue, blk_size, nbanks, clk_r, clk_w,  nreqs , avg_at, bank_cfl, exec_time\n\n");
    for (int t = 0; t < num_tables; t++ ) {
        for (int r = 0; r < controller_tables[t]->row; r++) {
            config.is_fcfs = controller_tables[t]->is_fcfs;
            config.max_waiting_queue_size = controller_tables[t]->max_waiting_queue_size[r];
            config.block_size = controller_tables[t]->block_size[r];
            config.num_of_banks = controller_tables[t]->num_of_banks[r];
            config.nclks_read = controller_tables[t]->dram_timing.nclks_read;
            config.nclks_write = controller_tables[t]->dram_timing.nclks_write;
            printf("%4u, %9u, %8u, %6u, %5u, %5u, ", config.is_fcfs, config.max_waiting_queue_size, 
                config.block_size, config.num_of_banks, config.nclks_read, config.nclks_write);

            // Initialize a CPU trace parser
            TraceParser *mem_trace = initTraceParser(argv[1]);

            // Initialize a Controller
            Controller *controller = initController(&config);
            // printf("%u\n", controller->bank_shift);
            // printf("%u\n", controller->bank_mask);

            uint64_t conflict_req = 0;

            uint64_t cycles = 0;
            uint64_t num_request = 0;

            bool stall = false;
            bool end = false;

            while (!end || ongoingPendingRequests(controller))
            {
                // if (num_request > 100) break;
                if (!end && !stall)
                {
                    end = !(getRequest(mem_trace));
                    ++num_request;
                }

                if (!end)
                {
                    stall = !(send(controller, mem_trace->cur_req));
                }
                tick(controller, &conflict_req);
                ++cycles;
            }
            printf("%5lu, %6lu, %8lu, ", num_request, controller->access_time/num_request, controller->bank_conficts);
            // printf("End Execution Time: ""%"PRIu64"\n", cycles);
            printf("""%"PRIu64"\n", cycles);
            
            free(controller->bank_status);
            free(controller->waiting_queue);
            free(controller->pending_queue);
            free(controller);
        }
        printf("-------------------------------------------------------------------\n");
    }
}
