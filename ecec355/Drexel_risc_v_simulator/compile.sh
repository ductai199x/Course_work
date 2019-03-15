#!/bin/bash

if [ "$1" == "" ]; then
	make -C src/single_cycle
	mv src/single_cycle/RVSim ./RVSim_single

elif [ "$1" == "-s" ]; then
	make -C src/single_cycle
	mv src/single_cycle/RVSim ./RVSim_single

elif [ "$1" == "-p" ]; then
	make -C src/pipeline
	mv src/pipeline/RVSim ./RVSim_pipeline

else
	echo "Unsupported architecture"
fi
