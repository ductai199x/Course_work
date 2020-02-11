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
    bool is_frfcfs;
    
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

    Controller_Table *fcfs_dram = malloc(sizeof(Controller_Table));
    fcfs_dram->dram_timing.nclks_read = 53;
    fcfs_dram->dram_timing.nclks_write = 53;
    unsigned fcfs_dram_max_waiting_queue_size[] = {64, 64, 64, 64};
    unsigned fcfs_dram_block_size[] = {128, 128, 128, 128};
    unsigned fcfs_dram_num_of_banks[] = {2, 4, 8, 16};
    fcfs_dram->max_waiting_queue_size = fcfs_dram_max_waiting_queue_size;
    fcfs_dram->block_size = fcfs_dram_block_size;
    fcfs_dram->num_of_banks = fcfs_dram_num_of_banks;
    fcfs_dram->is_frfcfs = false;
    fcfs_dram->row = sizeof(fcfs_dram_num_of_banks)/sizeof(fcfs_dram_num_of_banks[0]);

    Controller_Table *test = malloc(sizeof(Controller_Table));
    test->dram_timing.nclks_read = 53;
    test->dram_timing.nclks_write = 53;
    unsigned test_max_waiting_queue_size[] = {64};
    unsigned test_block_size[] = {128};
    unsigned test_num_of_banks[] = {16};
    test->max_waiting_queue_size = test_max_waiting_queue_size;
    test->block_size = test_block_size;
    test->num_of_banks = test_num_of_banks;
    test->is_frfcfs = true;
    test->row = sizeof(test_num_of_banks)/sizeof(test_num_of_banks[0]);

    Controller_Table *fcfs_pcm = malloc(sizeof(Controller_Table));
    fcfs_pcm->dram_timing.nclks_read = 57;
    fcfs_pcm->dram_timing.nclks_write = 162;
    unsigned fcfs_pcm_max_waiting_queue_size[] = {64, 64, 64, 64};
    unsigned fcfs_pcm_block_size[] = {128, 128, 128, 128};
    unsigned fcfs_pcm_num_of_banks[] = {2, 4, 8, 16};
    fcfs_pcm->max_waiting_queue_size = fcfs_pcm_max_waiting_queue_size;
    fcfs_pcm->block_size = fcfs_pcm_block_size;
    fcfs_pcm->num_of_banks = fcfs_pcm_num_of_banks;
    fcfs_pcm->is_frfcfs = false;
    fcfs_pcm->row = sizeof(fcfs_pcm_num_of_banks)/sizeof(fcfs_pcm_num_of_banks[0]);

    Controller_Table *frfcfs_dram = malloc(sizeof(Controller_Table));
    frfcfs_dram->dram_timing.nclks_read = 53;
    frfcfs_dram->dram_timing.nclks_write = 53;
    unsigned frfcfs_dram_max_waiting_queue_size[] = {64, 64, 64, 64};
    unsigned frfcfs_dram_block_size[] = {128, 128, 128, 128};
    unsigned frfcfs_dram_num_of_banks[] = {2, 4, 8, 16};
    frfcfs_dram->max_waiting_queue_size = frfcfs_dram_max_waiting_queue_size;
    frfcfs_dram->block_size = frfcfs_dram_block_size;
    frfcfs_dram->num_of_banks = frfcfs_dram_num_of_banks;
    frfcfs_dram->is_frfcfs = true;
    frfcfs_dram->row = sizeof(frfcfs_dram_num_of_banks)/sizeof(frfcfs_dram_num_of_banks[0]);

    Controller_Table *frfcfs_pcm = malloc(sizeof(Controller_Table));
    frfcfs_pcm->dram_timing.nclks_read = 57;
    frfcfs_pcm->dram_timing.nclks_write = 162;
    unsigned frfcfs_pcm_max_waiting_queue_size[] = {64, 64, 64, 64};
    unsigned frfcfs_pcm_block_size[] = {128, 128, 128, 128};
    unsigned frfcfs_pcm_num_of_banks[] = {2, 4, 8, 16};
    frfcfs_pcm->max_waiting_queue_size = frfcfs_pcm_max_waiting_queue_size;
    frfcfs_pcm->block_size = frfcfs_pcm_block_size;
    frfcfs_pcm->num_of_banks = frfcfs_pcm_num_of_banks;
    frfcfs_pcm->is_frfcfs = true;
    frfcfs_pcm->row = sizeof(frfcfs_pcm_num_of_banks)/sizeof(frfcfs_pcm_num_of_banks[0]);

    Controller_Table *controller_tables[] = { frfcfs_dram, frfcfs_pcm };
    // Controller_Table *controller_tables[] = { fcfs_dram, fcfs_pcm, frfcfs_dram, frfcfs_pcm };
    int num_tables = (int)( sizeof(controller_tables) / sizeof(controller_tables[0]) );

    printf("frfcfs, max_queue, blk_size, nbanks, clk_r, clk_w,  nreqs , avg_at, bank_cfl, exec_time\n");
    for (int t = 0; t < num_tables; t++ ) {
        for (int r = 0; r < controller_tables[t]->row; r++) {
            config.is_frfcfs = controller_tables[t]->is_frfcfs;
            config.max_waiting_queue_size = controller_tables[t]->max_waiting_queue_size[r];
            config.block_size = controller_tables[t]->block_size[r];
            config.num_of_banks = controller_tables[t]->num_of_banks[r];
            config.nclks_read = controller_tables[t]->dram_timing.nclks_read;
            config.nclks_write = controller_tables[t]->dram_timing.nclks_write;
            printf("%6u, %9u, %8u, %6u, %5u, %5u, ", config.is_frfcfs, config.max_waiting_queue_size, 
                config.block_size, config.num_of_banks, config.nclks_read, config.nclks_write);

            // Initialize a CPU trace parser
            TraceParser *mem_trace = initTraceParser(argv[1]);

            // Initialize a Controller
            Controller *controller = initController(&config);

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
            printf("%7lu, %6lu, %8lu, ", num_request, controller->access_time/num_request, controller->bank_conficts);
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
