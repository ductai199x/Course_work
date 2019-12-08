## Tai Duc Nguyen - ECEC 412 - 12/05/2019

# Extra Credit Questions 
*On "Runahead Execution: An Alternative to Very LargeInstruction Windows for Out-of-order Processors" by Onur Mutlu et al.*

- [Extra Credit Questions](#extra-credit-questions)
  - [Can you name two alternate strategies that we have covered in the course and address the same problem that this paper addresses?](#can-you-name-two-alternate-strategies-that-we-have-covered-in-the-course-and-address-the-same-problem-that-this-paper-addresses)
  - [Summarize in your own words the main idea of the paper in less than 90 words](#summarize-in-your-own-words-the-main-idea-of-the-paper-in-less-than-90-words)
  - [Summarize the key results of the paper in less than 90 words](#summarize-the-key-results-of-the-paper-in-less-than-90-words)
  

## Can you name two alternate strategies that we have covered in the course and address the same problem that this paper addresses?

The problem the paper addresses is reducing pipeline's stalls due to long latency instructions (with a finite instruction buffer). The other 2 techniques which attempt to resolve the issue of the pipeline stalling is:
- Prefetching:
  - Solves the problem by prefetching memory that is predicted to be useful in the near future
- Multithreading:
  - An additional thread can scan the program's instruction ahead of the main thread so that memory can be fetched prior to it being needed.


## Summarize in your own words the main idea of the paper in less than 90 words

The paper introduces a method called `runahead`, which allows the processor to "keep going" in the case of a cache-miss, instead of stalling. The processor enters "runahead mode" when it detects an L1 cache miss. Before entering this mode, the processor:
- records the address that causes the miss
- issues a fetch for the missing data in L2, L3 or main
- check-points register file (a snapshot of the reg file before entering the mode)



## Summarize the key results of the paper in less than 90 words

There are 3 main key results of the paper:
- The Runahead method outperforms systems without it in most benchmarks
- Hardware Prefetcher can be integrated with this method, and the system performance can improve/degrade depending on the performance of the Prefetcher.
- Runahead Execution allow systems with low number of instruction windows buffer to perform well, hence, optimize the system for both performance and energy consumption.
