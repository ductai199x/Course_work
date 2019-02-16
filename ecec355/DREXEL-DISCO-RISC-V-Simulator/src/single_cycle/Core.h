#ifndef __CORE_H__
#define __CORE_H__

#include <fstream>
#include <iostream>
#include <string>
#include <list>
#include <bitset>

#include "Instruction_Memory.h"
#include "Instruction.h"

using namespace std;

class Core
{
public:
	Core(const string &fname, ofstream *out);

	bool tick(); // FALSE means all the instructions are exhausted

	int id; // Each core has its own ID

	void printInstrs()
	{
		cout << "Core " << id << " : " << endl;

		instr_mem->printInstr();
	}


private:

	ofstream *out; // Output file

    uint64_t regFile[32];

    bool debug_mode;
    bool debug_to_stdout;

	unsigned long long int clk;

    uint8_t data_mem[4096]; 

	/*
		Group One: Design Components Here, an instruction memory has already been
		designed for you.
	*/
	long PC;

	Instruction_Memory *instr_mem;

	/*
		Group Two: Simulator Related
	*/
	list<Instruction> pending_queue;
    

    long int exec_branch(bitset<32> rs1, bitset<32> rs2, bitset<32> type, bitset<32> rd);
    void exec_r_op(bitset<32> rs1, bitset<32> rs2, bitset<32> type, bitset<32> rd);
    void exec_i_op(bitset<32> rs1, bitset<32> cnst, bitset<32> type, bitset<32> rd);
    void exec_load(bitset<32> rs1, bitset<32> offset, bitset<32> type, bitset<32> rd);
    void exec_store(bitset<32> rs1, bitset<32> offset, bitset<32> type, bitset<32> rd);
    long int exec_jal(bitset<32> offset, bitset<32> rd); 
    long int exec_jalr(bitset<32> rs1, bitset<32> offset, bitset<32> type, bitset<32> rd);
 
	void serve_pending_instrs();

	void printStats(list<Instruction>::iterator &ite);
};

#endif
