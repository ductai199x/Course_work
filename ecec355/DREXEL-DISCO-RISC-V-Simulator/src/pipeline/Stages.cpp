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
        if (PC >= instr_mem->last_addr())
        {
            // No instruction to run for next clock cycle since we have reached the last 
            // instruction.
            end = 1;
        }

        cout << "PC: " << PC << " ";	
        // Get Instruction
		Instruction &instruction = instr_mem->get_instruction(PC);

		// Increment PC
		// TODO, PC should be incremented or decremented based on instruction
        PC += PC_OFFSET;
        
		if_id_reg.valid = 1;

        if_id_reg.opcode = (instruction.instruction) & 127;
		if_id_reg.rd_index = (instruction.instruction >> 7) & 31;
        if_id_reg.funct3 = (instruction.instruction >> (7 + 5)) & 7;
		if_id_reg.rs1_index = (instruction.instruction >> (7 + 5 + 3)) & 31;
		if_id_reg.rs2_index = (instruction.instruction >> (7 + 5 + 3 + 5)) & 31;
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
			cout << "IF : " << instr->raw_instr << endl;	
		}
	}
}

void ID_Stage::hazard_detection()
{
	/*
	 * (1) EX/MEM.rd = ID/EX.rs1
	 * (2) EX/MEM.rd = ID/EX.rs2
	 * (3) MEM/WB.rd = ID/EX.rs1
	 * (4) MEM/WB.rd = ID/EX.rs2
	 * */

	if (ex_stage->ex_mem_reg.valid == 1)
	{
		if (ex_stage->ex_mem_reg.rd_index == id_ex_reg.rs1_index || 
			ex_stage->ex_mem_reg.rd_index == id_ex_reg.rs2_index)
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
		if (mem_stage->mem_wb_reg.rd_index == id_ex_reg.rs1_index || 
			mem_stage->mem_wb_reg.rd_index == id_ex_reg.rs2_index)
		{
			if_stage->stall = 1; // Fetching should not proceed.
			stall = 1; // ID should also stall.
			ex_stage->bubble = 1; // EX stage should not accept any new instructions
            mem_stage->bubble = 1; 

			instr->end_exe += 1; // The end execution time should be incremented by 1.
			return;
		}
	}
    else if (id_ex_reg.opcode == 99) {
        if_stage->stall = 1;

        instr->end_exe +=1;
        return;
    }

	if_stage->stall = 0; // No hazard found, fetching proceed.
	stall = 0; // No hazard found, ID stage proceed.
	ex_stage->bubble = 0; // No hazard found, execution proceed.
    mem_stage->bubble = 0;
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

    PC = &(if_stage->PC);
    PC_OFFSET = &(if_stage->PC_OFFSET);
	
	id_ex_reg.valid = if_stage->if_id_reg.valid;
	id_ex_reg.opcode = if_stage->if_id_reg.opcode;
	id_ex_reg.funct3 = if_stage->if_id_reg.funct3;
	id_ex_reg.funct7 = if_stage->if_id_reg.funct7;

    id_ex_reg.ex_op = NULL;
    id_ex_reg.mem_op = NULL;
    id_ex_reg.wb_op = NULL;

    // r-type operation
    if ( id_ex_reg.opcode == 51 ) {

        id_ex_reg.rs1_index = if_stage->if_id_reg.rs1_index;
        id_ex_reg.rs2_index = if_stage->if_id_reg.rs2_index;
        id_ex_reg.rd_index = if_stage->if_id_reg.rd_index; 

        id_ex_reg.a = regFile[id_ex_reg.rs1_index];
        id_ex_reg.b = regFile[id_ex_reg.rs2_index];
        
        id_ex_reg.wb_op = &WB_Stage::write_back;

        // add
        if ( id_ex_reg.funct3 == 0 && id_ex_reg.funct7 == 0 ) {
            id_ex_reg.ex_op = &EX_Stage::add;
        }
        // sub
        else if ( id_ex_reg.funct3 == 0 && id_ex_reg.funct7 == 32 ) {
            id_ex_reg.ex_op = &EX_Stage::sub;
        }
        // sll
        else if ( id_ex_reg.funct3 == 1 && id_ex_reg.funct7 == 0 ) {
            id_ex_reg.ex_op = &EX_Stage::shift_left;
        }
        // srl
        else if ( id_ex_reg.funct3 == 5 && id_ex_reg.funct7 == 0 ) {
            id_ex_reg.ex_op = &EX_Stage::shift_right;
        }
        // xor
        else if ( id_ex_reg.funct3 == 4 && id_ex_reg.funct7 == 0 ) {
            id_ex_reg.ex_op = &EX_Stage::_xor;
        }
        // or
        else if ( id_ex_reg.funct3 == 6 && id_ex_reg.funct7 == 0 ) {
            id_ex_reg.ex_op = &EX_Stage::_or;
        }
        // and
        else if ( id_ex_reg.funct3 == 7 && id_ex_reg.funct7 == 0 ) {
            id_ex_reg.ex_op = &EX_Stage::_and;
        }
    } 

    // i-type operation
    else if ( id_ex_reg.opcode == 19 ) {

        id_ex_reg.rs1_index = if_stage->if_id_reg.rs1_index;
        id_ex_reg.rs2_index = if_stage->if_id_reg.rs2_index;
        id_ex_reg.rd_index = if_stage->if_id_reg.rd_index; 
        id_ex_reg.imm = id_ex_reg.rs2_index | (id_ex_reg.funct7 << 5);
        id_ex_reg.imm = (id_ex_reg.imm & 2048) ? (id_ex_reg.imm-4096) : id_ex_reg.imm;

        id_ex_reg.a = regFile[id_ex_reg.rs1_index];
        id_ex_reg.b = id_ex_reg.imm;

        id_ex_reg.wb_op = &WB_Stage::write_back;
        
        // addi
        if ( id_ex_reg.funct3 == 0 ) {
            id_ex_reg.ex_op = &EX_Stage::add;
        }
        // slli
        else if ( id_ex_reg.funct3 == 1 ) {
            id_ex_reg.ex_op = &EX_Stage::shift_left;
        }
        // xori
        else if ( id_ex_reg.funct3 == 4 ) {
            id_ex_reg.ex_op = &EX_Stage::_xor;
        }
        // srli
        else if ( id_ex_reg.funct3 == 5 ) {
            id_ex_reg.ex_op = &EX_Stage::shift_right;
        }
        // ori
        else if ( id_ex_reg.funct3 == 6 ) {
            id_ex_reg.ex_op = &EX_Stage::_or;
        }
        // andi
        else if ( id_ex_reg.funct3 == 7 ) {
            id_ex_reg.ex_op = &EX_Stage::_and;
        }
    }

    // store operation
    else if ( id_ex_reg.opcode == 35 ) {

        id_ex_reg.rs1_index = if_stage->if_id_reg.rs1_index;
        id_ex_reg.rs2_index = if_stage->if_id_reg.rs2_index;
        id_ex_reg.rd_index = if_stage->if_id_reg.rd_index; 
        id_ex_reg.imm = if_stage->if_id_reg.rd_index | (if_stage->if_id_reg.funct7 << 5);
        id_ex_reg.imm = (id_ex_reg.imm & 2048) ? (id_ex_reg.imm-4096) : id_ex_reg.imm;

        id_ex_reg.a = regFile[id_ex_reg.rs1_index];
        id_ex_reg.b = id_ex_reg.imm; 

        // sd
        if ( id_ex_reg.funct3 == 3 ) {
            id_ex_reg.ex_op = &EX_Stage::add;
            id_ex_reg.mem_op = &MEM_Stage::store;
        }
    }

    // load operation
    else if ( id_ex_reg.opcode == 3 ) {

        id_ex_reg.rs1_index = if_stage->if_id_reg.rs1_index;
        id_ex_reg.rs2_index = if_stage->if_id_reg.rs2_index;
        id_ex_reg.rd_index = if_stage->if_id_reg.rd_index;
        id_ex_reg.imm = if_stage->if_id_reg.rs2_index | (if_stage->if_id_reg.funct7 << 5);
        id_ex_reg.imm = (id_ex_reg.imm & 2048) ? (id_ex_reg.imm-4096) : id_ex_reg.imm;

        id_ex_reg.a = regFile[id_ex_reg.rs1_index];
        id_ex_reg.b = id_ex_reg.imm;

        id_ex_reg.wb_op = &WB_Stage::write_back;

        // ld
        if ( id_ex_reg.funct3 == 3 ) {
            id_ex_reg.ex_op = &EX_Stage::add;
            id_ex_reg.mem_op = &MEM_Stage::load;
        }
    }

    // branch operation
    else if ( id_ex_reg.opcode == 99 ) {

        id_ex_reg.rs1_index = if_stage->if_id_reg.rs1_index;
        id_ex_reg.rs2_index = if_stage->if_id_reg.rs2_index;
        id_ex_reg.rd_index = if_stage->if_id_reg.rd_index;
        id_ex_reg.imm = if_stage->if_id_reg.rd_index | (if_stage->if_id_reg.funct7 << 5);
        id_ex_reg.imm = (id_ex_reg.imm & 2048) ? (id_ex_reg.imm-4097) : id_ex_reg.imm;

        id_ex_reg.a = *PC;
        id_ex_reg.b = id_ex_reg.imm;
        if_stage->end = 0;
        //if_stage->stall = 1;
        // beq
        if ( id_ex_reg.funct3 == 0 ) {
            if ( regFile[id_ex_reg.rs1_index] == regFile[id_ex_reg.rs2_index] )
                id_ex_reg.ex_op = &EX_Stage::move_pc_offset;
        }
        // bne
        else if ( id_ex_reg.funct3 == 0 ) {
            if ( regFile[id_ex_reg.rs1_index] != regFile[id_ex_reg.rs2_index] )
                id_ex_reg.ex_op = &EX_Stage::move_pc_offset;
        }
        // blt
        else if ( id_ex_reg.funct3 == 0 ) {
            if ( regFile[id_ex_reg.rs1_index] < regFile[id_ex_reg.rs2_index] )
                id_ex_reg.ex_op = &EX_Stage::move_pc_offset;
        }
        // bge
        else if ( id_ex_reg.funct3 == 0 ) {
            if ( regFile[id_ex_reg.rs1_index] >= regFile[id_ex_reg.rs2_index] )
                id_ex_reg.ex_op = &EX_Stage::move_pc_offset;
        }       
    }

    // jal operation
    else if ( id_ex_reg.opcode == 111 ) {
        
    }

    // jalr operation
    else if ( id_ex_reg.opcode == 103 ) {

    }

    else {

    }
	
    hazard_detection();

    if (DEBUG)
	{
        cout << "ID : " << instr->raw_instr;
		
		if (stall)
		{
			cout << " (stalled) ";
		}

		cout << endl;	
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
    
    if (DEBUG)
	{
		cout << "EX : " << instr->raw_instr << " -> ";	
	}

	ex_mem_reg.valid = id_stage->id_ex_reg.valid;
	id_stage->id_ex_reg.valid = 0; 
     
    ex_mem_reg.rd_index = id_stage->id_ex_reg.rd_index; 
    ex_mem_reg.rs1_index = id_stage->id_ex_reg.rs1_index; 
    ex_mem_reg.rs2_index = id_stage->id_ex_reg.rs2_index; 
    ex_mem_reg.mem_op = id_stage->id_ex_reg.mem_op;
    ex_mem_reg.wb_op = id_stage->id_ex_reg.wb_op;

    PC = id_stage->PC;
    PC_OFFSET = id_stage->PC_OFFSET;

    void (EX_Stage::*func)(long signed int, long signed int);
    func = id_stage->id_ex_reg.ex_op;

    (this->*func)(id_stage->id_ex_reg.a, id_stage->id_ex_reg.b);
    
    cout << "result: " << ex_mem_reg.result;

    cout << endl;
}

void MEM_Stage::tick()
{
    if (end == 1)
    {
        // Instructions are run out, do nothing.
        return;
    }
    
    if ( bubble == 1 ) {
        return;
    }

    
    if (!ex_stage->ex_mem_reg.valid)
    {
        // EX_MEM register is invalid, do nothing.
        return;
    }

	end = ex_stage->end; // end signal is propagated from IF stage

	instr = ex_stage->instr; // instruction pointer is also propagated from IF stage

    if (DEBUG)
	{
		cout << "MEM : " << instr->raw_instr << " -> ";	
	}
	
	mem_wb_reg.valid = ex_stage->ex_mem_reg.valid;
	ex_stage->ex_mem_reg.valid = 0;

	mem_wb_reg.rd_index = ex_stage->ex_mem_reg.rd_index;
	mem_wb_reg.rs1_index = ex_stage->ex_mem_reg.rs1_index;
	mem_wb_reg.rs2_index = ex_stage->ex_mem_reg.rs2_index;
    mem_wb_reg.result = ex_stage->ex_mem_reg.result;
    mem_wb_reg.wb_op = ex_stage->ex_mem_reg.wb_op;

    void (MEM_Stage::*func)(long unsigned int, long unsigned int);
    func = ex_stage->ex_mem_reg.mem_op;

    if ( func != NULL ) {
        if ( func == &MEM_Stage::store ) {
            (this->*func)(mem_wb_reg.rs2_index, mem_wb_reg.result);
            cout << "stored " << regFile[mem_wb_reg.rs2_index] << " into data_mem[" << mem_wb_reg.result << "]";
        }

        if ( func == &MEM_Stage::load ) {
            (this->*func)(mem_wb_reg.rd_index, mem_wb_reg.result);
            cout << "load " << mem_wb_reg.result << " from data_mem[" << ex_stage->ex_mem_reg.result << "]"; 
         }
    }
	
    cout << endl;
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
	
    if (DEBUG)
	{
		cout << "WB : " << instr->raw_instr << " -> ";
	}

	mem_stage->mem_wb_reg.valid = 0; 

    void (WB_Stage::*func)(int, long signed int);
    func = mem_stage->mem_wb_reg.wb_op;
    
    if ( func != NULL ) {
        (this->*func)(mem_stage->mem_wb_reg.rd_index, mem_stage->mem_wb_reg.result);
        cout << "wrote " << mem_stage->mem_wb_reg.result << " into x" << mem_stage->mem_wb_reg.rd_index; 
    }
		
    cout << endl;
}

void WB_Stage::write_back(int reg_index, long signed int value)
{
    regFile[reg_index] = value;
}

void MEM_Stage::store(long unsigned int value, long unsigned int addr)
{
    data_mem[addr] = regFile[value];
}

void MEM_Stage::load(long unsigned int value, long unsigned int addr)
{
    mem_wb_reg.result = data_mem[addr];
}

void EX_Stage::add(long signed int a, long signed int b)
{
    ex_mem_reg.result = a + b;
}

void EX_Stage::sub(long signed int a, long signed int b)
{
    ex_mem_reg.result = a - b;
}

void EX_Stage::shift_right(long signed int a, long signed int b)
{
    ex_mem_reg.result = a >> b;
}

void EX_Stage::shift_left(long signed int a, long signed int b)
{
    ex_mem_reg.result = a << b;
}

void EX_Stage::_xor(long signed int a, long signed int b)
{
    ex_mem_reg.result = a ^ b;
}

void EX_Stage::_or(long signed int a, long signed int b)
{
    ex_mem_reg.result = a | b;
}

void EX_Stage::_and(long signed int a, long signed int b)
{
    ex_mem_reg.result = a & b;
}

void EX_Stage::move_pc_offset(long signed int a, long signed int b)
{
    *PC = a + b - 4;
    ex_mem_reg.result = b;
    id_stage->if_stage->stall = 0;
}
 
 
