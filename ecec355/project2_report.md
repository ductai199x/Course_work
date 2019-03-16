Tai Duc Nguyen - ECEC 355 - 03/15/2019

# PROJECT 2 REPORT

## Pipeline Implementation
- The pipeline’s main components are the 5 stages objects, which are connected together through many “register” c++ structs: if_id_reg, id_ex_reg, ex_mem_reg and mem_wb_reg 
- `if_id_reg` includes 
    - All the extracted fields from the instruction's code
- `id_ex_reg` includes:
    - All the fields from `if_id_reg`
    - 3 more functions pointers (`ex_op, mem_op, wb_op`), which defines the operations for later stages
    - 3 more variables (`a, b, imm`) so that the Execution Stage can take these given number and do the specified operation in `ex_op`
- `ex_mem_reg` includes:
    - All the fields from `if_id_reg`
    - 2 functions pointer (`mem_op, wb_op`) inherited from `id_ex_reg`
    - A `result` field to store the result from the operation specified in `ex_op`
- `mem_wb_reg` includes:
    - All fields from `if_id_reg`
    - 1 function pointer (`wb_op`) inherited from `ex_mem_reg`
    - A `result` field to store the result from the operation specified in `mem_op`
- The operations are defined within the corresponding Stage class
- The pipeline works on the premise that the output of the current stage is the input of the next stage. Hence, by specifying a `function` and that function's `arguments` in Stage A, Stage B (following A) can just take those information and perform the operation. This method makes the code base very easy to read and understand

## Hazard detection unit
### Without forwarding
- Data Hazard:
    - The unit always stalls untill the value is available from `WB Stage`, then the pipeline is resumed. This will add 3 clock cycles to any data dependencies
- Control Hazard
    - The unit treats control hazard same as it treats data hazrd. It will stalls until the condition of the branch instruction is determined. Then it will resume the pipeline. This will add 3 clock cycles to any control dependencies

```
if (ex_stage->ex_mem_reg.valid == 1)
	{
		if (ex_stage->ex_mem_reg.rd_index == id_ex_reg.rs1_index || 
			ex_stage->ex_mem_reg.rd_index == id_ex_reg.rs2_index)
		{
			if_stage->stall = 1; 
			stall = 1; 
			ex_stage->bubble = 1;
			instr->end_exe = 1; 
			return;
		}
	}
	else if (mem_stage->mem_wb_reg.valid == 1)
	{
		if (mem_stage->mem_wb_reg.rd_index == id_ex_reg.rs1_index || 
			mem_stage->mem_wb_reg.rd_index == id_ex_reg.rs2_index)
		{
			if_stage->stall = 1;
			stall = 1;
			ex_stage->bubble = 1;
            mem_stage->bubble = 1; 
			instr->end_exe = 1;
			return;
		}
}
```

### With forwarding
- Data Hazard:
    - The unit will do forwarding, eliminating the need to stall the pipeline for all instructions (no stalling)
- Control Hazard:
    - With forwarding, the unit will forward the result to the branch instruction, hence, require only 1 additional bubble in the pipeline.

```
if (ex_stage->ex_mem_reg.valid == 1)
{
    // Load
    if (id_ex_reg.opcode == 3) {
        if (ex_stage->ex_mem_reg.rd_index == id_ex_reg.rs1_index) {
            id_ex_reg.a = ex_stage->ex_mem_reg.result;
        }
        if (mem_stage->mem_wb_reg.rd_index == id_ex_reg.rs1_index) {
            id_ex_reg.a = mem_stage->mem_wb_reg.result;
        }
        return;
    }
    // Store
    else if (id_ex_reg.opcode == 35) {
        if (ex_stage->ex_mem_reg.rd_index == id_ex_reg.rs1_index) {
            id_ex_reg.a = ex_stage->ex_mem_reg.result;
        }
        if (mem_stage->mem_wb_reg.rd_index == id_ex_reg.rs1_index) {
            id_ex_reg.a = mem_stage->mem_wb_reg.result;
        }

        return;
    }
    // Branch
    else if (id_ex_reg.opcode == 99) {
        if_stage->stall = 1;
        stall = 1;
        ex_stage->bubble = 1;
    }
    // Other
    else {
        if (ex_stage->ex_mem_reg.opcode == 3){ return; }
        if (ex_stage->ex_mem_reg.opcode == 35){ return; }
        if (mem_stage->mem_wb_reg.rd_index == id_ex_reg.rs1_index) {
            id_ex_reg.a = mem_stage->mem_wb_reg.result;
        }
        if (mem_stage->mem_wb_reg.rd_index == id_ex_reg.rs2_index) {
            id_ex_reg.b = mem_stage->mem_wb_reg.result;
        }
        if (ex_stage->ex_mem_reg.rd_index == id_ex_reg.rs1_index) {
            id_ex_reg.a = ex_stage->ex_mem_reg.result;
        }
        if (ex_stage->ex_mem_reg.rd_index == id_ex_reg.rs2_index) {
            id_ex_reg.b = ex_stage->ex_mem_reg.result;
        }
        return;
    }
}

```

# Forwarding Unit
- For different type of instructions:
    - Load/Store: if `rs1 == rd` of `ex_stage` or `mem_stage`, then copy such value in ex or mem stage to variable `a` in `id_ex_reg`
    - Branch: Insert 1 bubble so that the number will be available from the ex stage 