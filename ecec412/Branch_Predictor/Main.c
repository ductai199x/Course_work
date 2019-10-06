#include "Trace.h"
#include "Branch_Predictor.h"

extern TraceParser *initTraceParser(const char * trace_file);
extern bool getInstruction(TraceParser *cpu_trace);

extern Branch_Predictor *initBranchPredictor(BP_Config *config);
extern bool predict(Branch_Predictor *branch_predictor, Instruction *instr, BP_Config *config);

typedef struct BP_TABLE {
    int row;
    unsigned* local_predictor_size;
    unsigned* local_counter_bits;
    unsigned* local_history_table_size;
    unsigned* global_predictor_size;
    unsigned* choice_predictor_size;
    char* name;
}BP_TABLE;

int main(int argc, const char *argv[])
{	
    if (argc != 2)
    {
        printf("Usage: %s %s\n", argv[0], "<trace-file>");

        return 0;
    }

    BP_Config config = (BP_Config) {
        .bp_type = "",
        .local_predictor_size = 2048,
        .local_history_table_size = 2048,
        .global_predictor_size = 8192,
        .choice_predictor_size = 8192,
        .local_counter_bits = 2,
        .global_counter_bits = 2,
        .choice_counter_bits = 2
    };

    
    BP_TABLE *twobitlocal_table = malloc(sizeof(BP_TABLE));
    twobitlocal_table->row = 7;
    twobitlocal_table->name = "twobitlocal";
    unsigned lps_arr[] = {2048, 2048, 4096, 8192, 16384, 32768, 65536};
    unsigned lcb_arr[] = {1, 2, 2, 2, 2, 2, 2};
    twobitlocal_table->local_predictor_size = lps_arr;
    twobitlocal_table->local_counter_bits = lcb_arr;

    BP_TABLE *gshare_table = malloc(sizeof(BP_TABLE));
    gshare_table->row = 7;
    gshare_table->name = "gshare";
    gshare_table->local_predictor_size = lps_arr;
    gshare_table->local_counter_bits = lcb_arr;

    BP_TABLE *tournament_table = malloc(sizeof(BP_TABLE));
    tournament_table->row = 3;
    tournament_table->name = "tournament";
    unsigned lhts_arr[] = {2048, 2048, 4096};
    unsigned gps_arr[] = {8192, 8192, 16384};
    unsigned cps_arr[] = {8192, 8192, 16384};
    tournament_table->local_history_table_size = lhts_arr;
    tournament_table->global_predictor_size = gps_arr;
    tournament_table->choice_predictor_size = cps_arr;

    BP_TABLE *bp_tables[] = { twobitlocal_table, tournament_table };

    int num_tables = (int)( sizeof(bp_tables) / sizeof(bp_tables[0]) );

    for (int t = 0; t < num_tables; t++ ) {
        config.bp_type = bp_tables[t]->name;
        for (int r = 0; r < bp_tables[t]->row; r++) {
            if (!strcmp(bp_tables[t]->name, "twobitlocal")) {
                config.local_predictor_size = bp_tables[t]->local_predictor_size[r];
                config.local_counter_bits = bp_tables[t]->local_counter_bits[r];
                printf("%s, %u, %u, ", config.bp_type, config.local_predictor_size, config.local_counter_bits);
            }
            else if (!strcmp(bp_tables[t]->name, "tournament")) {
                config.local_history_table_size = bp_tables[t]->local_history_table_size[r];
                config.global_predictor_size = bp_tables[t]->global_predictor_size[r];
                config.choice_predictor_size = bp_tables[t]->choice_predictor_size[r];
                printf("%s, %u, %u, %u, ", config.bp_type, config.local_history_table_size, config.global_predictor_size, config.choice_predictor_size);
            }
            // Initialize a CPU trace parser
            TraceParser *cpu_trace = initTraceParser(argv[1]);
            // Initialize a branch predictor
            Branch_Predictor *branch_predictor = initBranchPredictor(&config);

            // Running the trace
            uint64_t num_of_instructions = 0;
            uint64_t num_of_branches = 0;
            uint64_t num_of_correct_predictions = 0;
            uint64_t num_of_incorrect_predictions = 0;

            

            while (getInstruction(cpu_trace))
            {
                // We are only interested in BRANCH instruction
                if (cpu_trace->cur_instr->instr_type == BRANCH)
                {
                    ++num_of_branches;

                    if (predict(branch_predictor, cpu_trace->cur_instr, &config))
                    {
                        ++num_of_correct_predictions;
                    }
                    else
                    {
                        ++num_of_incorrect_predictions;
                    }
                }
                ++num_of_instructions;
            }

            // printf("Number of instructions: %"PRIu64"\n", num_of_instructions);
            // printf("Number of branches: %"PRIu64"\n", num_of_branches);
            // printf("Number of correct predictions: %"PRIu64"\n", num_of_correct_predictions);
            // printf("Number of incorrect predictions: %"PRIu64"\n", num_of_incorrect_predictions);

            float performance = (float)num_of_correct_predictions / (float)num_of_branches * 100;
            // printf("Predictor Correctness: %f%%\n", performance);
            printf("%"PRIu64", %"PRIu64", %"PRIu64", %"PRIu64", %f%%\n", num_of_instructions, num_of_branches, num_of_correct_predictions, num_of_incorrect_predictions, performance);
        }
        printf("\n----------------------------------\n");
    }
}
