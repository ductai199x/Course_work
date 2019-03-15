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

class IF_Stage
{
public:
        IF_Stage(const string &fname, Core* core) : 
                instr_mem(new Instruction_Memory(fname)),
                core(core),
                PC(0),
                PC_OFFSET(4),
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
        
        long PC;
        long signed int PC_OFFSET;

        Instruction_Memory *instr_mem;

        struct Register
        {
            int valid; // Is content inside register valid?
            int opcode;
            int funct3;
            int funct7;
            int rd_index;
            int rs1_index;
            int rs2_index;
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

        void hazard_detection();

        list<Instruction>::iterator instr; // Points to the instruction currently in the stage

        int stall; // Is the stage stalled?
	    int end; // All instructions are exhausted?

        uint64_t* regFile;
        uint8_t* data_mem;

        long* PC;
        long signed int* PC_OFFSET;

        IF_Stage *if_stage;
        EX_Stage *ex_stage;
	    MEM_Stage *mem_stage;
        WB_Stage *wb_stage;

        struct Register
        {
            int valid; // Is content inside register valid?
            void (EX_Stage::*ex_op)(long signed int, long signed int);
            void (MEM_Stage::*mem_op)(long unsigned int, long unsigned int);
            void (WB_Stage::*wb_op)(int, long signed int); 
            long signed int a;
            long signed int b;
            int opcode;
            int funct3;
            int funct7;
            int rd_index;
            int rs1_index;
            int rs2_index;
            long signed int imm;
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

        ID_Stage *id_stage;

        long* PC;
        long signed int* PC_OFFSET;

        void add(long signed int a, long signed int b);
        void sub(long signed int a, long signed int b);
        void shift_right(long signed int a, long signed int b);        
        void shift_left(long signed int a, long signed int b);
        void _xor(long signed int a, long signed int b);
        void _or(long signed int a, long signed int b);
        void _and(long signed int a, long signed int b);
        void move_pc_offset(long signed int a, long signed int b);

        struct Register
        {
            int valid; // Is content inside register valid?
            void (EX_Stage::*ex_op)(long signed int, long signed int);
            void (MEM_Stage::*mem_op)(long unsigned int, long unsigned int);
            void (WB_Stage::*wb_op)(int, long signed int); 
            int opcode;
            int funct3;
            int funct7;
            int rd_index;
            int rs1_index;
            int rs2_index;

            long signed int result;
            long signed int imm;
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
        int bubble;
	
        uint64_t* regFile;
        uint8_t* data_mem;

        EX_Stage *ex_stage;

        void store(long unsigned int value, long unsigned int addr);
        void load(long unsigned int value, long unsigned int addr);

        struct Register
        {
            int valid; // Is content inside register valid?
            void (EX_Stage::*ex_op)(long signed int, long signed int);
            void (MEM_Stage::*mem_op)(long unsigned int, long unsigned int);
            void (WB_Stage::*wb_op)(int, long signed int); 
            int opcode;
            int funct3;
            int funct7;
            int rd_index;
            int rs1_index;
            int rs2_index;
            long signed int result;
            long signed int imm;
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
