#!/bin/bash

# Array of input sizes (2^16, 2^18, ..., 2^28)
inputSizes=(65536 262144 1048576 4194304 16777216 67108864)

# Array of CUDA threads (64, 128, ..., 4096)
mpiProcs=(128)

inputTypes=("Sorted" "ReverseSorted" "Random" "Perturbed")

# Iterate over each input size
for size in "${inputSizes[@]}"; do
    for proc in "${mpiProcs[@]}"; do
        for inputType in "${inputTypes[@]}"; do
        # Execute the command with the current size and thread count
            sbatch mpi.grace_job $size $proc $inputType
        done
    done
done