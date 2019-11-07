#include "Core.h"
#include <string>
#include <iomanip>
#include <bitset>
#include <cstdint>
#include <cstring>

Core::Core(const string &fname, ofstream *out) : out(out),
						clk(0),
						PC(0),
						instr_mem(new Instruction_Memory(fname))
{
    regFile[0] = 0;
    regFile[2] = 4095;
    debug_mode = true;
    debug_to_stdout = false; // all debug statment will be printed to *out file if this is false
    uint8_t mat[16] = {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2};
    uint8_t output[4] = {1,1,1,1};

    memcpy(data_mem, mat, sizeof(mat));
    memcpy(data_mem+16, output, sizeof(output));

    regFile[10] = 0;
    regFile[11] = 16;
}


long int Core::exec_branch(bitset<32> rs1, bitset<32> rs2, bitset<32> type, bitset<32> rd)
{
    int reg1 = rs1.to_ulong();
    int reg2 = rs2.to_ulong();
    long int pc_offset = (rd[11] == 1) ? rd.to_ulong()-4097 : rd.to_ulong();
    cout << pc_offset << endl;


    long int ret = 4;
    int funct = type.to_ulong();

    if ( funct == 0 ) {
        if ( regFile[reg1] == regFile[reg2] )
        ret = pc_offset;
    }
    else if ( funct == 1 ) {
        if ( regFile[reg1] != regFile[reg2] )
        ret = pc_offset;
    }
    else if ( funct == 4 ) {
        if ( regFile[reg1] < regFile[reg2] )
        ret = pc_offset;
    }
    else if ( funct == 5 ) {
        if ( regFile[reg1] >= regFile[reg2] )
        ret = pc_offset;
    }
    else {
        cout << "non-implemented branch operation no. " << funct << " !" << endl;
    }

    if ( debug_mode ) {
        if ( debug_to_stdout )
            cout << "cur PC: " << PC << " offset: " << ret;
        else
            *out << "cur PC: " << PC << " offset: " << ret;
    }
    return ret;
}

void Core::exec_r_op(bitset<32> rs1, bitset<32> rs2, bitset<32> type, bitset<32> rd)
{
    int reg1 = rs1.to_ulong();
    int reg2 = rs2.to_ulong();
    int regD = rd.to_ulong();
    long int funct = type.to_ulong();

    if ( funct == 0 ) {
        regFile[regD] = regFile[reg1] + regFile[reg2];
    }
    else if ( funct == 512 ) {
        regFile[regD] = regFile[reg1] - regFile[reg2];
    }
    else if ( funct == 7 ) {
        regFile[regD] = regFile[reg1] & regFile[reg2];
    }
    else if ( funct == 6 ) {
        regFile[regD] = regFile[reg1] | regFile[reg2];
    }
    else if ( funct == 4 ) {
        regFile[regD] = regFile[reg1] ^ regFile[reg2];
    }
    else if ( funct == 5 ) {
        regFile[regD] = regFile[reg1] >> regFile[reg2];
    }
    else if ( funct == 1 ) {
        regFile[regD] = regFile[reg1] << regFile[reg2];
    }
    else if ( funct == 8 ) {
        regFile[regD] = regFile[reg1] * regFile[reg2];
    }
    else {
        cout << "non implemented r-operation!" << endl;
    }

    if ( debug_mode ) {
        if ( debug_to_stdout )
            cout << "rs1(x" << reg1 << "): " << regFile[reg1] << " rs2(x" << reg2 << "): " << regFile[reg2] << " regD(x" << regD << "): " << regFile[regD];
        else
            *out << "rs1(x" << reg1 << "): " << regFile[reg1] << " rs2(x" << reg2 << "): " << regFile[reg2] << " regD(x" << regD << "): " << regFile[regD];
    }
}

void Core::exec_i_op(bitset<32> rs1, bitset<32> cnst, bitset<32> type, bitset<32> rd)
{
    int reg1 = rs1.to_ulong();
    long int constant = (cnst[11] == 1) ? cnst.to_ulong()-4096 : cnst.to_ulong();
    int regD = rd.to_ulong();
    long int funct = type.to_ulong();

    if ( funct == 0 ) {
        regFile[regD] = regFile[reg1] + constant;
    }
    else if ( funct == 7 ) {
        regFile[regD] = regFile[reg1] & constant;
    }
    else if ( funct == 6 ) {
        regFile[regD] = regFile[reg1] | constant;
    }
    else if ( funct == 4 ) {
        regFile[regD] = regFile[reg1] ^ constant;
    }
    else if ( funct == 5 ) {
        regFile[regD] = regFile[reg1] >> constant;
    }
    else if ( funct == 1 ) {
        regFile[regD] = regFile[reg1] << constant;
    }
    else {
        cout << "non-implemented i-operation!" << endl;
    }

    if ( debug_mode ) {
        if ( debug_to_stdout )
            cout << "rs1(x" << reg1 << "): " << regFile[reg1] << " const: " << constant << " regD(x" << regD << "): " << regFile[regD];
        else
            *out << "rs1(x" << reg1 << "): " << regFile[reg1] << " const: " << constant << " regD(x" << regD << "): " << regFile[regD];
    }
}

void Core::exec_load(bitset<32> rs1, bitset<32> offset, bitset<32> type, bitset<32> rd)
{
    int reg1 = rs1.to_ulong();
    int mem_offset = offset.to_ulong();
    int regD = rd.to_ulong();
    long int funct = type.to_ulong();

    int mem_ptr = 0;

    if ( funct == 3 ) {
        regFile[regD] = data_mem[regFile[reg1] + mem_offset];
    }
    else if ( funct == 2 ) {
        regFile[regD] = data_mem[regFile[reg1] + mem_offset];
    }
    else if ( funct == 4 ) {
        regFile[regD] = data_mem[regFile[reg1] + mem_offset];
    }
    else {
        cout << "non-implemented load-operation!" << endl;
    }

    if ( debug_mode ) {
        if ( debug_to_stdout )
            cout << "rs1(x" << reg1 << "): " << regFile[reg1] << " offset: " << mem_offset << " regD(x" << regD << "): " << regFile[regD];
        else
            *out << "rs1(x" << reg1 << "): " << regFile[reg1] << " offset: " << mem_offset << " regD(x" << regD << "): " << regFile[regD];
    }
}

void Core::exec_store(bitset<32> rs1, bitset<32> offset, bitset<32> type, bitset<32> rd)
{
    int regD = rs1.to_ulong();
    int mem_offset = offset.to_ulong();
    int reg1 = rd.to_ulong();
    long int funct = type.to_ulong();

    int mem_ptr = 0;

    if ( funct == 3 ) {
        data_mem[regFile[regD] + mem_offset] = regFile[reg1];
    }
    else if ( funct == 2 ) {
        data_mem[regFile[regD] + mem_offset] = regFile[reg1];
    }
    else {
        cout << "non-implemented store-operation!" << endl;
    }

    if ( debug_mode ) {
        if ( debug_to_stdout )
            cout << "rs1(x" << reg1 << "): " << regFile[reg1] << " offset: " << mem_offset << " reg2(x" << regD << "): " << regFile[regD];
        else
            *out << "rs1(x" << reg1 << "): " << regFile[reg1] << " offset: " << mem_offset << " reg2(x" << regD << "): " << regFile[regD];
    }
}

long int Core::exec_jal(bitset<32> offset, bitset<32> rd)
{
    long int mem_offset;
    int regD = rd.to_ulong();


    regFile[regD] = PC + 4;
    mem_offset = (offset[19] == 1) ? offset.to_ulong()-2097152 : offset.to_ulong();

    if ( debug_mode ) {
        if ( debug_to_stdout )
            cout << "[jal] " << "offset: " << mem_offset;
        else
            *out << "[jal] " << "offset: " << mem_offset;
    }

    return mem_offset;

}

long int Core::exec_jalr(bitset<32> rs1, bitset<32> offset, bitset<32> type, bitset<32> rd)
{
    int reg1 = rs1.to_ulong();
    long int mem_offset;
    int regD = rd.to_ulong();
    int funct = type.to_ulong();

    regFile[regD] = PC + 4;
    mem_offset = ( regFile[reg1] + ( (offset[11] == 1) ? offset.to_ulong()-4097 : offset.to_ulong() )) & 0xfffffffe;

    if ( debug_mode ) {
        if ( debug_to_stdout )
            cout << "[jalr] " << "offset: " << mem_offset;
        else
            *out << "[jalr] " << "offset: " << mem_offset;
    }

    return mem_offset;

}

bool Core::tick()
{
	/*
		Step One: Serving pending instructions
	*/
	if (pending_queue.size() > 0)
	{
		serve_pending_instrs();
	}

	/*
		Step Two: Where simulation happens
	*/
	if (PC <= instr_mem->last_addr())
	{
		// Get Instruction
		Instruction &instruction = instr_mem->get_instruction(PC);

        bitset<32> ins(instruction.instruction);
        bitset<32> opcode(std::string(  "00000000000000000000000001111111"));
        bitset<32> rd(std::string(      "00000000000000000000111110000000"));
        bitset<32> funct3(std::string(  "00000000000000000111000000000000"));
        bitset<32> rs1(std::string(     "00000000000011111000000000000000"));
        bitset<32> rs2(std::string(     "00000001111100000000000000000000"));
        bitset<32> funct7(std::string(  "11111110000000000000000000000000"));

        opcode   = ins & opcode;
        rd       = ins & rd;
        funct3   = ins & funct3;
        rs1      = ins & rs1;
        rs2      = ins & rs2;
        funct7   = ins & funct7;

        bitset<32> branch(std::string(  "1100011"));
        bitset<32> jalr(std::string(    "1100111"));
        bitset<32> jal(std::string(     "1101111"));
        bitset<32> i_op(std::string(    "0010011"));
        bitset<32> load(std::string(    "0000011"));
        bitset<32> store(std::string(   "0100011"));
        bitset<32> r_op(std::string(    "0110011"));

        if ( debug_mode ) {
            if ( debug_to_stdout )
                cout << "[ " << left << setw(20) << instruction.raw_instr << "] PC: " << left << setw(4) << PC << " => ";
            else
                *out << "[ " << left << setw(20) << instruction.raw_instr << "] PC: " << left << setw(4) << PC << " => ";
        }

        int PC_OFFSET = 4;
        if ( opcode == branch ) {
            PC_OFFSET = exec_branch(rs1>>15, rs2>>20, funct3>>12, (funct7>>20)|(rd>>7));
        }

        else if ( opcode == jal ) {
            PC_OFFSET = exec_jal((funct7>>20)|(rs2>>20), rd>>7);
        }

        else if ( opcode == jalr ) {
            PC = exec_jalr(rs1>>15, (funct7>>20)|(rs2>>20), funct3>>12, rd>>7);
            PC_OFFSET = 0;
        }

        else if ( opcode == i_op ) {
            exec_i_op(rs1>>15, (funct7>>20)|(rs2>>20), funct3>>12, rd>>7);
        }

        else if ( opcode == load ) {
            exec_load(rs1>>15, (funct7>>20)|(rs2>>20), funct3>>12, rd>>7);
        }

        else if ( opcode == store ) {
            exec_store(rs1>>15, (funct7>>20)|(rd>>7), funct3>>12, rs2>>20);
        }

        else if ( opcode == r_op ) {
            exec_r_op(rs1>>15, rs2>>20, (funct7>>22)|(funct3>>12), rd>>7);
        }

        else {
            cout << "opcode not implemented: " << ins << endl;
        }

		PC += PC_OFFSET;

		instruction.begin_exe = clk;
		// Single-cycle always takes one clock cycle to complete
		instruction.end_exe = clk + 1;

		pending_queue.push_back(instruction);

        if ( debug_mode ) {
            if ( debug_to_stdout )
                cout << endl;
            else
                *out << endl;
        }
	}

	clk++;

	/*
		Step Four: Should we shut down simulator
	*/
	if (pending_queue.size() == 0)
	{
		return false;
	}
	else
	{
		return true;
	}
}

void Core::serve_pending_instrs()
{
	list<Instruction>::iterator instr = pending_queue.begin();

	if (instr->end_exe <= clk)
	{
		//printStats(instr);

		pending_queue.erase(instr);
	}
}

void Core::printStats(list<Instruction>::iterator &ite)
{
	*out << ite->raw_instr << " => ";
	*out << "Core ID: " << id << "; ";
	*out << "Begin Exe: " << ite->begin_exe << "; ";
	*out << "End Exe: " << ite->end_exe << endl;
}
