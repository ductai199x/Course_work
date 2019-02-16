#include "Core.h"

Core::Core(const string &fname, ofstream *out) : out(out), 
						clk(0) 
{
	regFile[0] = 0;
    regFile[2] = 4095;

    if_stage = (new IF_Stage(fname, this));	
	id_stage = (new ID_Stage(regFile, data_mem));
	ex_stage = (new EX_Stage());
	mem_stage = (new MEM_Stage(regFile, data_mem));
	wb_stage = (new WB_Stage(regFile));

	wb_stage->mem_stage = mem_stage;
	wb_stage->id_stage = id_stage;

	mem_stage->ex_stage = ex_stage;
	
	ex_stage->id_stage = id_stage;
	
	id_stage->if_stage = if_stage;
	id_stage->ex_stage = ex_stage;
	id_stage->mem_stage = mem_stage;

    id_stage->operation_arr[0] = &EX_Stage::add;
    id_stage->operation_arr[1] = &EX_Stage::sub;
    id_stage->operation_arr[2] = &EX_Stage::shift_right;
    id_stage->operation_arr[3] = &EX_Stage::shift_left;
    id_stage->operation_arr[4] = &EX_Stage::_xor;
    id_stage->operation_arr[5] = &EX_Stage::_or;
    id_stage->operation_arr[6] = &EX_Stage::_and;
    id_stage->operation_arr[7] = &EX_Stage::calc_addr;
  
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
	if (DEBUG)
	{
		cout << "clk: " << clk << " : ";
	}

	wb_stage->tick();
	mem_stage->tick();
	ex_stage->tick();
	id_stage->tick();	
	if_stage->tick();
	
	if (DEBUG)
	{
		cout << endl;
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
		printStats(instr);
		
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

