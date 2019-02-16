#ifndef __STAGES_H__
#define __STAGES_H__

#include <fstream>
#include <iostream>
#include <string>
#include <list>

#include "Core.h"

#include "Instruction_Memory.h"
#include "Instruction.h"

using namespace std;

class Core;

class IF_Stage;
class ID_Stage;
class EX_Stage;
class MEM_Stage;
class WB_Stage;

/*
 * The following classes are for demonstration only. You should modify it based on the requirements.
 * */
class IF_Stage
{
public:
        IF_Stage(const string &fname, Core* core) : 
                instr_mem(new Instruction_Memory(fname)),
                core(core),
                PC(0),
                stall(0),
                end(0)
	    {
                // Initially, IF/ID Register is invalid.
                if_id_reg.valid = 0;
	    }

        void tick();

        list<Instruction>::iterator instr; // Points to the instruction currently in the stage

        int stall; // Is the stage stalled?
        int end; // All instructions are exhausted?

        Core *core;
        
        // TODO, Design components of IF stage here
        
        long PC;

        Instruction_Memory *instr_mem;

        
        // TODO, define your IF/ID register here.
        
        /*
	    * Here shows the prototype of an in-complete IF/ID register. You should 
	    * extend it further to get a complete IF/ID register.
	    * */
        struct Register
        {
            int valid; // Is content inside register valid?

            int opcode;
            int funct3;
            int funct7;
            int rd_index;
            int rs_1_index;
            int rs_2_index;
        };
        Register if_id_reg;
};

class ID_Stage
{
public:
        ID_Stage(uint64_t* regFile, uint8_t* data_mem) : 
                stall(0), 
                end(0),
                regFile(regFile),
                data_mem(data_mem)
        {
                id_ex_reg.valid = 0;
        }

        void tick();

        /*
         * Hazard detection unit: stall ID and IF stages, meanwhile, insert bubbles to
         * EX stage.
         * */
        void hazard_detection();

        list<Instruction>::iterator instr; // Points to the instruction currently in the stage

        int stall; // Is the stage stalled?
	    int end; // All instructions are exhausted?

        uint64_t* regFile;
        uint8_t* data_mem;

        void (EX_Stage::*operation_arr[8])(long signed int*, long signed int*);
       
	    // Hazard detection unit needs access to IF and EX stage.
        IF_Stage *if_stage;
        EX_Stage *ex_stage;
	    MEM_Stage *mem_stage;

        struct Register
        {
            int valid; // Is content inside register valid?
            int stages;
            void (EX_Stage::*operation)(long signed int*, long signed int*);
            long signed int a;
            long signed int b;
            int opcode;
            int funct3;
            int funct7;
            int rd_index;
            int rs_1_index;
            int rs_2_index;

    	};
        Register id_ex_reg;
};

class EX_Stage
{
public:
        EX_Stage() : bubble(0), end(0)
	    {
                ex_mem_reg.valid = 0;
        }

        void tick();

        list<Instruction>::iterator instr; // Points to the instruction currently in the stage
        
        int bubble; // A bubble is inserted?
        int end; // All instructions are exhausted?

        /*
         * Related Class
         * */
        ID_Stage *id_stage;

        void add(long signed int *a, long signed int *b);
        void sub(long signed int *a, long signed int *b);
        void shift_right(long signed int *a, long signed int *b);        
        void shift_left(long signed int *a, long signed int *b);
        void _xor(long signed int *a, long signed int *b);
        void _or(long signed int *a, long signed int *b);
        void _and(long signed int *a, long signed int *b);
        void calc_addr(long signed int *a, long signed int *b);

        struct Register
        {
            int valid; // Is content inside register valid?
            int stages;
            int opcode;
            int funct3;
            int funct7;
            int rd_index;
            int rs_1_index;
            int rs_2_index;

            long signed int result;
        };
        Register ex_mem_reg;
};

class MEM_Stage
{
public:
        MEM_Stage(uint64_t *regFile, uint8_t *data_mem) : 
                end(0),
                regFile(regFile),
                data_mem(data_mem)
        {
                mem_wb_reg.valid = 0;
        }

        void tick();

        list<Instruction>::iterator instr; // Points to the instruction currently in the stage
        
        int end; // All instructions are exhausted?
	
        uint64_t* regFile;
        uint8_t* data_mem;


        EX_Stage *ex_stage;

        struct Register
        {
            int valid; // Is content inside register valid?
            int stages;
            int opcode;
            int funct3;
            int funct7;
            int rd_index;
            int rs_1_index;
            int rs_2_index;

            long signed int result;
        };
        Register mem_wb_reg;
};

class WB_Stage
{
public:
        WB_Stage(uint64_t *regFile) : 
                end(0),
                regFile(regFile)
        {

        }

        void tick();

        list<Instruction>::iterator instr; // Points to the instruction currently in the stage
        
        int end; // All instructions are exhausted?
        uint64_t *regFile;

        MEM_Stage *mem_stage;
        ID_Stage *id_stage;

        void write_back(int reg_index, long signed int value);

};

#endif
