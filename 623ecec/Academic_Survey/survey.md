## Tai Duc Nguyen - ECEC 623 - 03/08/2020

## Academic Survey

_This document includes the sumary of different academic papers which focus on Memory Controler (MC) optimizations_

1. [CROW: A Low-Cost Substrate for Improving DRAM Performance, Energy Efficiency, and Reliability](#crow-a-low-cost-substrate-for-improving-dram-performance-energy-efficiency-and-reliability)
2. [ATLAS: A Scalable and High-Performance Scheduling Algorithm for Multiple Memory Controllers](#atlas-a-scalable-and-high-performance-scheduling-algorithm-for-multiple-memory-controllers)
3. [High-Performance and Energy-Efficient Memory Scheduler Design for Heterogeneous Systems](#high-performance-and-energy-efficient-memory-scheduler-design-for-heterogeneous-systems)
4. [PARBS](#parbs)
5. [BLISS](#bliss)
6. [EDEN: Enabling Energy-Efficient, High-Performance Deep Neural Network Inference Using Approximate DRAM](#eden-enabling-energy-efficient-high-performance-deep-neural-network-inference-using-approximate-dram)
7. [A Memory Controller with Row Buffer Locality Awareness for Hybrid Memory Systems](#a-memory-controller-with-row-buffer-locality-awareness-for-hybrid-memory-systems)

<!-- - Research Team:
- Type of Memory:
- Research Focus:
- Simulators:
- Benchmarks: -->

# CROW: A Low-Cost Substrate for Improving DRAM Performance, Energy Efficiency, and Reliability

- Research Team:
  - **ETH Zurich, CMU** - Hasan Hassan et. al.
- Type of Memory:
  - DRAM
- Research Focus:
  - CROW (CopyRow) DRAM is a new DRAM chip design + optimization in the MC that can improve the performance, energy eff, and reliability. This technology enables 2 implementations, which can be combined in 1 system, such as: CROW-cache and CROW-ref. CROW-cache implements a way to reduce latency of the most-recently-accessed rows by caching the data of those "regular rows" in "copy rows" (of the same bank), so that when data is accessed, both rows are activated at the same time, making charge-sharing more efficient, and hence, reduce latency. CROW-ref, on the other hand, uses efficient profiling to identify "weak rows" (rows that have low refresh interval), and mapped them to "strong copy rows" so that the refresh interval of the entire chip is much longer (2x-4x). These 2 CROW methods can be done with minimal change: CROW-table in MC, 2 new commands (ACT-t & ACT-c) and 1 independent decoder (for the copyrows).
- Simulators:
  - SPICE (open sourced): github.com/CMU-SAFARI/CROW
  - CACTI (open sourced): www.hpl.hp.com/techreports/2009/HPL-2009-85.html
- Benchmarks:
  - SPEC CPU2006 www.spec.org/cpu2006/
  - TPC www.tpc.org/
  - STREAM github.com/jeffhammond/STREAM
  - MediaBench mathstat.slu.edu/~fritts/mediabench/
  - 2 synthetic applications: 1) random 2) streaming (not open sourced)

# ATLAS: A Scalable and High-Performance Scheduling Algorithm for Multiple Memory Controllers

- Research Team:
  - **CMU** - Yoongu Kim et. al.
- Type of Memory:
  - DRAM
- Research Focus:
  - ATLAS (Adaptive per-Thread Least Attained-Service memory scheduling) is a new memory scheduling technique which improves system throughput in multi-processing-core-system without requring significant coordination among MCs. The technique takes both short-term and long-term thread behavior into account when ranking them. For short-term, it follows the LAS (least-attained service) rule, prioritizing threads that were least served in the last period. For long-term, it calculates the total attained service (from begining to time period $i$):
    - $total_{AS} = \alpha\times total_{AS-1} + \frac{(1-\alpha)}{thread\_weight}*AS_i$
  - Then the thread with the least $total_{AS}$ will be ranked highest.
  - Also, outstanding requests that have been waiting in the queue for more than T cycles will be prioritized over all others.
- Simulators:
  - In-house cycle-precise x86 CMP simulator (not open sourced)
- Benchmarks:
  - SPEC CPU2006 (410.bwaves, 416.gamess, 434.zeusmp are not included)

# High-Performance and Energy-Efficient Memory Scheduler Design for Heterogeneous Systems

- Research Team:
  - **CMU, AMD Research, Intel Labs, Facebook, ETH Zurich** - Rachata Ausavarungnirun et. al
- Type of Memory:
  - DRAM
- Research Focus:
  - In heterogeneous systems where CPUs and GPU (on the same die), for example, share the same off-chip DRAM, the requests from the GPU can interfere with those from CPUs. Hence, the paper proposes SMS (Staged Memory Scheduler), which decouples the primary MC tasks into simpler structures. MC tasks are generally described as:
    - Detection of basic intra-application memory characteristics (row buffer locality)
    - Prioritization across applications & enforcement of policies
    - Low-level command scheduling, enforcement of DRAM timing constraints, and resolution of resource conflicts
  - Instead of all 3 tasks being performed at the same time, SMS divides 3 tasks into 3 stages:
    - **Batch Formation Stage**: Form batches for each CPU cores and GPU. Batches are ready when all requests (except first one) access the same DRAM row
    - **Batch Scheduling Stage**: This stage can be implemented with different existing scheduling policies (FRFCFS, PARBS, ATLAS), but the author propose one where SJF (Shortest Job First) policy is used with a probability $p$. If $p > 0.5$, then SJF is applied more often: applications with fewer outstanding requests are prioritized (CPU over GPU). If $p < 0.5$, then GPU is prioritized over CPU.
    - **DRAM Command Scheduler**: This stage is where the controller pop individual requests from batches received from the previous stage and schedule them according to timing constraints.
- Simulators:
  - In-house cycle-accurate simulator (not open sourced)
- Benchmarks:
  - 16 benchmarks from SPEC CPU2006 + 1 GPU application.

# PARBS

- Research Team:
- Type of Memory:
- Research Focus:
- Simulators:
- Benchmarks:

# BLISS

- Research Team:
- Type of Memory:
- Research Focus:
- Simulators:
- Benchmarks:

# EDEN: Enabling Energy-Efficient, High-Performance Deep Neural Network Inference Using Approximate DRAM

- Research Team:
  - **ETH Zurich** - Skanda Koppula et. al.
- Type of Memory:
  - DRAM
- Research Focus:
  - 
- Simulators:
- Benchmarks:


# A Memory Controller with Row Buffer Locality Awareness for Hybrid Memory Systems

https://arxiv.org/pdf/1804.11040.pdf
- Research Team:
- Type of Memory:
- Research Focus:
- Simulators:
- Benchmarks: