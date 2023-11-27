#!/bin/bash

# Define the arrays for inputSizes and num_procs
inputSizes=(65536 262144 1048576 4194304 16777216 67108864 268435456)
numProcs=(512 1024)

# Loop through each combination of inputSize and num_procs
for size in "${inputSizes[@]}"; do
    for proc in "${numProcs[@]}"; do
        # Run the sbatch command with the current combination and a fixed inputType of 0
        sbatch mpi.grace_job "$proc" "$size" 3
    done
done
