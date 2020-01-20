#include "Branch_Predictor.h"

#define MAX_WEIGHT 127
#define MIN_WEIGHT -128

const unsigned instShiftAmt = 2; // Number of bits to shift a PC by

Branch_Predictor *initBranchPredictor(BP_Config *config)
{
    Branch_Predictor *branch_predictor = (Branch_Predictor *)malloc(sizeof(Branch_Predictor));

    if (!strcmp(config->bp_type, "twobitlocal")) {
        branch_predictor->local_predictor_sets = config->local_predictor_size;
        assert(checkPowerofTwo(branch_predictor->local_predictor_sets));

        branch_predictor->index_mask = branch_predictor->local_predictor_sets - 1;

        // Initialize sat counters
        branch_predictor->local_counters =
            (Sat_Counter *)malloc(branch_predictor->local_predictor_sets * sizeof(Sat_Counter));

        int i = 0;
        for (i; i < branch_predictor->local_predictor_sets; i++)
        {
            initSatCounter(&(branch_predictor->local_counters[i]), config->local_counter_bits);
        }
    }

    else if (!strcmp(config->bp_type, "gshare")) {
        branch_predictor->global_predictor_sets = config->global_predictor_size;
        assert(checkPowerofTwo(branch_predictor->global_predictor_sets));

        branch_predictor->global_history_mask = branch_predictor->global_predictor_sets - 1;

        // Initialize sat counters
        branch_predictor->global_counters = 
            (Sat_Counter *)malloc(config->global_predictor_size * sizeof(Sat_Counter));

        for (int i = 0; i < config->global_predictor_size; i++)
        {
            initSatCounter(&(branch_predictor->global_counters[i]), config->global_counter_bits);
        }
        
        branch_predictor->global_history_reg = 0;
    }

    else if (!strcmp(config->bp_type, "tournament")) {
        assert(checkPowerofTwo(config->local_predictor_size));
        assert(checkPowerofTwo(config->local_history_table_size));
        assert(checkPowerofTwo(config->global_predictor_size));
        assert(checkPowerofTwo(config->choice_predictor_size));
        assert(config->global_predictor_size == config->choice_predictor_size);

        branch_predictor->local_predictor_size = config->local_predictor_size;
        branch_predictor->local_history_table_size = config->local_history_table_size;
        branch_predictor->global_predictor_size = config->global_predictor_size;
        branch_predictor->choice_predictor_size = config->choice_predictor_size;
    
        // Initialize local counters 
        branch_predictor->local_counters =
            (Sat_Counter *)malloc(config->local_predictor_size * sizeof(Sat_Counter));

        int i = 0;
        for (i; i < config->local_predictor_size; i++)
        {
            initSatCounter(&(branch_predictor->local_counters[i]), config->local_counter_bits);
        }

        branch_predictor->local_predictor_mask = config->local_predictor_size - 1;

        // Initialize local history table
        branch_predictor->local_history_table = 
            (unsigned *)malloc(config->local_history_table_size * sizeof(unsigned));

        for (i = 0; i < config->local_history_table_size; i++)
        {
            branch_predictor->local_history_table[i] = 0;
        }

        branch_predictor->local_history_table_mask = config->local_history_table_size - 1;

        // Initialize global counters
        branch_predictor->global_counters = 
            (Sat_Counter *)malloc(config->global_predictor_size * sizeof(Sat_Counter));

        for (i = 0; i < config->global_predictor_size; i++)
        {
            initSatCounter(&(branch_predictor->global_counters[i]), config->global_counter_bits);
        }

        branch_predictor->global_history_mask = config->global_predictor_size - 1;

        // Initialize choice counters
        branch_predictor->choice_counters = 
            (Sat_Counter *)malloc(config->choice_predictor_size * sizeof(Sat_Counter));

        for (i = 0; i < config->choice_predictor_size; i++)
        {
            initSatCounter(&(branch_predictor->choice_counters[i]), config->choice_counter_bits);
        }

        branch_predictor->choice_history_mask = config->choice_predictor_size - 1;

        // global history register
        branch_predictor->global_history = 0;

        // We assume choice predictor size is always equal to global predictor size.
        branch_predictor->history_register_mask = config->choice_predictor_size - 1;
    }

    else if (!strcmp(config->bp_type, "perceptron")) {
        srand(0);
        branch_predictor->total_budget = config->total_budget;
        assert(checkPowerofTwo(branch_predictor->total_budget));

        branch_predictor->n_weights = config->n_weights;
        branch_predictor->bits_in_weight = config->bits_in_weight;

        branch_predictor->global_history_reg = 0;

        branch_predictor->n_perceptrons = branch_predictor->total_budget/((branch_predictor->n_weights + 1) * branch_predictor->bits_in_weight);

        branch_predictor->perc_table = (Perceptron *)malloc(sizeof(Perceptron)*branch_predictor->n_perceptrons);

        // Initialize neural network
        for (int i = 0; i < branch_predictor->n_perceptrons; i++) {
            branch_predictor->perc_table[i].weights = (int8_t *)malloc(sizeof(int8_t)*branch_predictor->n_weights);
            for (int j = 0; j < branch_predictor->n_weights; j++) {
                branch_predictor->perc_table[i].weights[j] = 0;
            }
        }
    }

    return branch_predictor;
}

// sat counter functions
inline void initSatCounter(Sat_Counter *sat_counter, unsigned counter_bits)
{
    sat_counter->counter_bits = counter_bits;
    sat_counter->counter = 0;
    sat_counter->max_val = (1 << counter_bits) - 1;
}

inline void incrementCounter(Sat_Counter *sat_counter)
{
    if (sat_counter->counter < sat_counter->max_val)
    {
        ++sat_counter->counter;
    }
}

inline void decrementCounter(Sat_Counter *sat_counter)
{
    if (sat_counter->counter > 0) 
    {
        --sat_counter->counter;
    }
}

// Branch Predictor functions
bool predict(Branch_Predictor *branch_predictor, Instruction *instr, BP_Config *config)
{
    uint64_t branch_address = instr->PC;

    if (!strcmp(config->bp_type, "twobitlocal")) { 
        // Step one, get prediction
        unsigned local_index = getIndex(branch_address, 
                                        branch_predictor->index_mask);

        bool prediction = getPrediction(&(branch_predictor->local_counters[local_index]));

        // Step two, update counter
        if (instr->taken)
        {
            // printf("Correct: %u -> ", branch_predictor->local_counters[local_index].counter);
            incrementCounter(&(branch_predictor->local_counters[local_index]));
            // printf("%u\n", branch_predictor->local_counters[local_index].counter);
        }
        else
        {
            // printf("Incorrect: %u -> ", branch_predictor->local_counters[local_index].counter);
            decrementCounter(&(branch_predictor->local_counters[local_index]));
            // printf("%u\n", branch_predictor->local_counters[local_index].counter);
        }

        return prediction == instr->taken;
    }

    else if (!strcmp(config->bp_type, "gshare")) {
        unsigned global_predictor_idx = (branch_predictor->global_history_reg ^ branch_address) & branch_predictor->global_history_mask;
        
        bool global_prediction = getPrediction(&(branch_predictor->global_counters[global_predictor_idx]));

        if (instr->taken)
        {
            incrementCounter(&(branch_predictor->global_counters[global_predictor_idx]));
        }
        else
        {
            decrementCounter(&(branch_predictor->global_counters[global_predictor_idx]));
        }

        branch_predictor->global_history_reg = branch_predictor->global_history_reg << 1 | instr->taken;

        return global_prediction == instr->taken;
    }

    else if (!strcmp(config->bp_type, "tournament")) {
        // Step one, get local prediction.
        unsigned local_history_table_idx = getIndex(branch_address,
                                            branch_predictor->local_history_table_mask);
        
        unsigned local_predictor_idx = 
            branch_predictor->local_history_table[local_history_table_idx] & 
            branch_predictor->local_predictor_mask;

        bool local_prediction = 
            getPrediction(&(branch_predictor->local_counters[local_predictor_idx]));

        // Step two, get global prediction.
        unsigned global_predictor_idx = 
            branch_predictor->global_history & branch_predictor->global_history_mask;

        bool global_prediction = 
            getPrediction(&(branch_predictor->global_counters[global_predictor_idx]));

        // Step three, get choice prediction.
        unsigned choice_predictor_idx = 
            branch_predictor->global_history & branch_predictor->choice_history_mask;

        bool choice_prediction = 
            getPrediction(&(branch_predictor->choice_counters[choice_predictor_idx]));


        // Step four, final prediction.
        bool final_prediction;
        if (choice_prediction)
        {
            final_prediction = global_prediction;
        }
        else
        {
            final_prediction = local_prediction;
        }

        bool prediction_correct = final_prediction == instr->taken;
        // Step five, update counters
        if (local_prediction != global_prediction)
        {
            if (local_prediction == instr->taken)
            {
                // Should be more favorable towards local predictor.
                decrementCounter(&(branch_predictor->choice_counters[choice_predictor_idx]));
            }
            else if (global_prediction == instr->taken)
            {
                // Should be more favorable towards global predictor.
                incrementCounter(&(branch_predictor->choice_counters[choice_predictor_idx]));
            }
        }

        if (instr->taken)
        {
            incrementCounter(&(branch_predictor->global_counters[global_predictor_idx]));
            incrementCounter(&(branch_predictor->local_counters[local_predictor_idx]));
        }
        else
        {
            decrementCounter(&(branch_predictor->global_counters[global_predictor_idx]));
            decrementCounter(&(branch_predictor->local_counters[local_predictor_idx]));
        }

        // Step six, update global history register
        branch_predictor->global_history = branch_predictor->global_history << 1 | instr->taken;
        branch_predictor->local_history_table[local_history_table_idx] = branch_predictor->local_history_table[local_history_table_idx] << 1 | instr->taken;
        // exit(0);
        //
        return prediction_correct;
    }

    else if (!strcmp(config->bp_type, "perceptron")) {
        unsigned perceptron_idx = branch_address % branch_predictor->n_perceptrons;

        int y = getPercOutput(&(branch_predictor->perc_table[perceptron_idx]), branch_predictor->global_history_reg, branch_predictor->n_weights);
        
        if (((y >= 0 ? 1 : 0) != instr->taken) || abs(y) < (branch_predictor->n_weights*1.93 + 14)) {
            updateWeights(&(branch_predictor->perc_table[perceptron_idx]), branch_predictor->global_history_reg, y, instr->taken, branch_predictor->n_weights);
        }

        branch_predictor->global_history_reg = branch_predictor->global_history_reg << 1 | instr->taken;

        return (y >= 0 ? 1 : 0) == instr->taken;
    }
}

inline unsigned getIndex(uint64_t branch_addr, unsigned index_mask)
{
    return (branch_addr >> instShiftAmt) & index_mask;
}

inline bool getPrediction(Sat_Counter *sat_counter)
{
    uint8_t counter = sat_counter->counter;
    unsigned counter_bits = sat_counter->counter_bits;

    // MSB determins the direction
    return (counter >> (counter_bits - 1));
}

inline void updateWeights(Perceptron *perc, unsigned ght, int y, int outcome, unsigned n_weights)
{
	if (perc->bias > MIN_WEIGHT && perc->bias < MAX_WEIGHT) {
		perc->bias += (outcome ? 1 : -1);
	}
   
    uint64_t ght_mask = 1;
    for (int i = 0; i < n_weights; i++) {
        if (perc->weights[i] <= MIN_WEIGHT && perc->weights[i] >= MAX_WEIGHT)
            continue;
        if( ( ( (ght & ght_mask) != 0) && (outcome == 1) ) || ( ((ght & ght_mask) == 0) && (outcome == 0) ) ) 
            perc->weights[i] += 1;
        else
            perc->weights[i] -= 1;

        ght_mask = ght_mask << 1;
    }
}

inline int getPercOutput(Perceptron *perc, unsigned ght, unsigned n_weights)
{
    int output_sum = perc->bias;

    uint64_t ght_mask = 1;
    for (int i = 0; i < n_weights; i++) {
        output_sum += ((ght & ght_mask) == 0 ? -1 : 1)*perc->weights[i];
        ght_mask = ght_mask << 1;
    }

    return output_sum;
}

int checkPowerofTwo(unsigned x)
{
    //checks whether a number is zero or not
    if (x == 0)
    {
        return 0;
    }

    //true till x is not equal to 1
    while( x != 1)
    {
        //checks whether a number is divisible by 2
        if(x % 2 != 0)
        {
            return 0;
        }
        x /= 2;
    }
    return 1;
}