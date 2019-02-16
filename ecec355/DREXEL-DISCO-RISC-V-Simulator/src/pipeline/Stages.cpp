#include "Stages.h"

void IF_Stage::tick()
{
	if (end == 1)
	{
		// No instruction to run, return.
		return;
	}

	

	if (stall == 0)
	{
        if (PC == instr_mem->last_addr())
        {
            // No instruction to run for next clock cycle since we have reached the last 
            // instruction.
            end = 1;
        }	
        // Get Instruction
		Instruction &instruction = instr_mem->get_instruction(PC);

		// Increment PC
		// TODO, PC should be incremented or decremented based on instruction
		PC += 4;

		if_id_reg.valid = 1;

		// For demonstration, I assume all instructions are R-type.

        if_id_reg.opcode = (instruction.instruction) & 127;
		if_id_reg.rd_index = (instruction.instruction >> 7) & 31;
        if_id_reg.funct3 = (instruction.instruction >> (7 + 5)) & 7;
		if_id_reg.rs_1_index = (instruction.instruction >> (7 + 5 + 3)) & 31;
		if_id_reg.rs_2_index = (instruction.instruction >> (7 + 5 + 3 + 5)) & 31;
        if_id_reg.funct7 = (instruction.instruction >> (7 + 5 + 3 + 5 + 5)) & 127;

		instruction.begin_exe = core->clk;

		// Initialize end execution time to 5 clock cycles but adjust it
		// if a stall detected.
		instruction.end_exe = core->clk + 4;

		// Push instruction object into queue
		(core->pending_queue).push_back(instruction);
		instr = (core->pending_queue).end();
		instr--;

		if (DEBUG)
		{
			cout << "IF : " << instr->raw_instr << " | ";	
		}
	}
}

void ID_Stage::hazard_detection()
{
	/*
	 * TODO, fix me, please modify this function in order to get a complete detection unit.
	 * For demonstration, I assume all instructions are R-type (WB is always set).
	 * */

	/*
	 * (1) EX/MEM.rd = ID/EX.rs1
	 * (2) EX/MEM.rd = ID/EX.rs2
	 * (3) MEM/WB.rd = ID/EX.rs1
	 * (4) MEM/WB.rd = ID/EX.rs2
	 * */
    
    // cout << "ex.rd: " << ex_stage->ex_mem_reg.rd_index << " ex.rs1: " << ex_stage->ex_mem_reg.rs_1_index << " ex.rs2: " << ex_stage->ex_mem_reg.rs_2_index << " id.rs1: " << id_ex_reg.rs_1_index << " id.rs2: " << id_ex_reg.rs_2_index << endl;

	if (ex_stage->ex_mem_reg.valid == 1)
	{
		if (ex_stage->ex_mem_reg.rd_index == id_ex_reg.rs_1_index || 
			ex_stage->ex_mem_reg.rd_index == id_ex_reg.rs_2_index)
		{
			if_stage->stall = 1; // Fetching should not proceed.
			stall = 1; // ID should also stall.
			ex_stage->bubble = 1; // EX stage should not accept any new instructions

			instr->end_exe += 1; // The end execution time should be incremented by 1.
			return;
		}
	}
	else if (mem_stage->mem_wb_reg.valid == 1)
	{
		if (mem_stage->mem_wb_reg.rd_index == id_ex_reg.rs_1_index || 
			mem_stage->mem_wb_reg.rd_index == id_ex_reg.rs_2_index)
		{
			if_stage->stall = 1; // Fetching should not proceed.
			stall = 1; // ID should also stall.
			ex_stage->bubble = 1; // EX stage should not accept any new instructions

			instr->end_exe += 1; // The end execution time should be incremented by 1.
			return;
		}
	}

	if_stage->stall = 0; // No hazard found, fetching proceed.
	stall = 0; // No hazard found, ID stage proceed.
	ex_stage->bubble = 0; // No hazard found, execution proceed.
}

void ID_Stage::tick()
{
    if (end == 1 && stall == 0)
    {
        // Instructions are run out, do nothing.
        return;
    }

    if (!if_stage->if_id_reg.valid)
    {
        // IF_ID register is invalid, do nothing.
        return;
    }
	
	end = if_stage->end; // end signal is propagated from IF stage
	
	instr = if_stage->instr; // instruction pointer is also propagated from IF stage
	
	id_ex_reg.valid = if_stage->if_id_reg.valid;

	id_ex_reg.opcode = if_stage->if_id_reg.opcode;
	id_ex_reg.rd_index = if_stage->if_id_reg.rd_index;
	id_ex_reg.funct3 = if_stage->if_id_reg.funct3;
    id_ex_reg.rs_1_index = if_stage->if_id_reg.rs_1_index;
	id_ex_reg.rs_2_index = if_stage->if_id_reg.rs_2_index;
	id_ex_reg.funct7 = if_stage->if_id_reg.funct7;

    long signed int rs1 = regFile[id_ex_reg.rs_1_index];
    long signed int rs2 = regFile[id_ex_reg.rs_2_index];
    long signed int rd  = regFile[id_ex_reg.rd_index];
    long signed int imm;

    // r-type operation
    if ( id_ex_reg.opcode == 51 ) {

        id_ex_reg.a = rs1;
        id_ex_reg.b = rs2;
        id_ex_reg.stages = 23;
        
        // add
        if ( id_ex_reg.funct3 == 0 && id_ex_reg.funct7 == 0 ) {
            id_ex_reg.operation = operation_arr[0];
        }
        // sub
        if ( id_ex_reg.funct3 == 0 && id_ex_reg.funct7 == 32 ) {
            id_ex_reg.operation = operation_arr[1];
        }
        // sll
        if ( id_ex_reg.funct3 == 1 && id_ex_reg.funct7 == 0 ) {
            id_ex_reg.operation = operation_arr[3];
        }
        // srl
        if ( id_ex_reg.funct3 == 5 && id_ex_reg.funct7 == 0 ) {
            id_ex_reg.operation = operation_arr[2];
        }
        // xor
        if ( id_ex_reg.funct3 == 4 && id_ex_reg.funct7 == 0 ) {
            id_ex_reg.operation = operation_arr[4];
        }
        // or
        if ( id_ex_reg.funct3 == 6 && id_ex_reg.funct7 == 0 ) {
            id_ex_reg.operation = operation_arr[5];
        }
        // and
        if ( id_ex_reg.funct3 == 7 && id_ex_reg.funct7 == 0 ) {
            id_ex_reg.operation = operation_arr[6];
        }
    } 

    // i-type operation
    if ( id_ex_reg.opcode == 19 ) {

        id_ex_reg.a = rs1;
        imm = id_ex_reg.rs_2_index | (id_ex_reg.funct7 << 5);
        id_ex_reg.b = imm;
        id_ex_reg.stages = 23;

        // addi
        if ( id_ex_reg.funct3 == 0 ) {
            id_ex_reg.operation = operation_arr[0];
        }
        // slli
        if ( id_ex_reg.funct3 == 1 ) {
            id_ex_reg.operation = operation_arr[3];
        }
        // xori
        if ( id_ex_reg.funct3 == 4 ) {
            id_ex_reg.operation = operation_arr[4];
        }
        // srli
        if ( id_ex_reg.funct3 == 5 ) {
            id_ex_reg.operation = operation_arr[2];
        }
        // ori
        if ( id_ex_reg.funct3 == 6 ) {
            id_ex_reg.operation = operation_arr[5];
        }
        // andi
        if ( id_ex_reg.funct3 == 7 ) {
            id_ex_reg.operation = operation_arr[6];
        }
    }

    // store operation
    if ( id_ex_reg.opcode == 35 ) {

        // sd
        if ( id_ex_reg.funct3 == 3 ) {

        }
    }

    // load operation
    if ( id_ex_reg.opcode == 3 ) {

        // ld
        if ( id_ex_reg.funct3 == 3 ) {

        }
    }

    // branch operation
    if ( id_ex_reg.opcode == 99 ) {

        // beq
        if ( id_ex_reg.funct3 == 0 ) {

        }
        // bne
        if ( id_ex_reg.funct3 == 0 ) {

        }
        // blt
        if ( id_ex_reg.funct3 == 0 ) {

        }
        // bge
        if ( id_ex_reg.funct3 == 0 ) {

        }
    }

    // jal operation
    if ( id_ex_reg.opcode == 111 ) {
        
    }

    // jalr operation
    if ( id_ex_reg.opcode == 103 ) {

    }
    
	
    hazard_detection();


    if (DEBUG)
	{
        cout << "ID : " << instr->raw_instr;
		
		if (stall)
		{
			cout << " (stalled) ";
		}

		cout << " | ";	
	}
}

void EX_Stage::tick()
{
	if (bubble == 1)
    {
        // A bubble is inserted, do nothing.
        return;
    }

    if (end == 1)
    {
        // Instructions are run out, do nothing.
        return;
    }

    if (!id_stage->id_ex_reg.valid)
    {
        // ID_EX register is invalid, do nothing.
        return;
    }

	end = id_stage->end; // end signal is propagated from IF stage

	instr = id_stage->instr; // instruction pointer is also propagated from IF stage

	ex_mem_reg.valid = id_stage->id_ex_reg.valid;
	id_stage->id_ex_reg.valid = 0; 
   
    ex_mem_reg.rd_index = id_stage->id_ex_reg.rd_index; 
    ex_mem_reg.rs_1_index = id_stage->id_ex_reg.rs_1_index; 
    ex_mem_reg.rs_2_index = id_stage->id_ex_reg.rs_2_index; 

    // I only allow any unique instruction to be read only 
	// once in order to increase simulator performance.

    void (EX_Stage::*func)(long signed int*, long signed int*);
    func = id_stage->id_ex_reg.operation;

    (this->*func)(&id_stage->id_ex_reg.a, &id_stage->id_ex_reg.b);

    ex_mem_reg.result = id_stage->id_ex_reg.a;
	
	if (DEBUG)
	{
		cout << "EX : " << instr->raw_instr << " | ";	
	}
}

void MEM_Stage::tick()
{
    if (end == 1)
    {
        // Instructions are run out, do nothing.
        return;
    }

    // Propagate `result` to WB in case MEM stage is skipped
    mem_wb_reg.result = ex_stage->ex_mem_reg.result;

    if (!ex_stage->ex_mem_reg.valid)
    {
        // EX_MEM register is invalid, do nothing.
        return;
    }

	end = ex_stage->end; // end signal is propagated from IF stage

	instr = ex_stage->instr; // instruction pointer is also propagated from IF stage
	
	mem_wb_reg.valid = ex_stage->ex_mem_reg.valid;
	ex_stage->ex_mem_reg.valid = 0;
    
	mem_wb_reg.rd_index = ex_stage->ex_mem_reg.rd_index;
	mem_wb_reg.rs_1_index = ex_stage->ex_mem_reg.rs_1_index;
	mem_wb_reg.rs_2_index = ex_stage->ex_mem_reg.rs_2_index;

	if (DEBUG)
	{
		cout << "MEM : " << instr->raw_instr << " | ";	
	}
}

void WB_Stage::tick()
{
    if (end == 1)
    {
        // Instructions are run out, do nothing.
        return;
    }

    if (!mem_stage->mem_wb_reg.valid)
    {
        // MEM_WB register is invalid, do nothing.
        return;
    }

	end = mem_stage->end; // end signal is propagated from IF stage
	
	instr = mem_stage->instr; // instruction pointer is also propagated from IF stage
	
	mem_stage->mem_wb_reg.valid = 0; 

    write_back(mem_stage->mem_wb_reg.rd_index, mem_stage->mem_wb_reg.result);

	if (DEBUG)
	{
		cout << "WB : " << instr->raw_instr << " | ";
	}	
}

void WB_Stage::write_back(int reg_index, long signed int value)
{
    regFile[reg_index] = value;
    cout << "wrote " << regFile[reg_index] << " to x" << reg_index << endl;
}

void EX_Stage::add(long signed int* a, long signed int* b)
{
    *a = *a + *b;
}

void EX_Stage::sub(long signed int* a, long signed int* b)
{
    *a = *a - *b;
}

void EX_Stage::shift_right(long signed int* a, long signed int* b)
{
    *a = *a >> *b;
}

void EX_Stage::shift_left(long signed int* a, long signed int* b)
{
    *a = *a << *b;
}

void EX_Stage::_xor(long signed int* a, long signed int* b)
{
    *a = *a ^ *b;
}

void EX_Stage::_or(long signed int* a, long signed int* b)
{
    *a = *a | *b;
}

void EX_Stage::_and(long signed int* a, long signed int* b)
{
    *a = *a & *b;
}

void EX_Stage::calc_addr(long signed int* a, long signed int* b)
{

}
 
 
