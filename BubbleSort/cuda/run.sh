#!/bin/bash

# Array of input sizes (2^16, 2^18, ..., 2^28)
# inputSizes=(65536 262144 1048576 4194304 16777216 67108864 268435456)
inputSizes=(1048576)

# Array of CUDA threads (64, 128, ..., 4096)
# cudaThreads=(64 128 256 512 1024)
cudaThreads=(512 1024 2048 4096)

inputTypes=("Sorted" "ReverseSorted" "Random" "Perturbed")

# Iterate over each input size
for size in "${inputSizes[@]}"; do
    for threads in "${cudaThreads[@]}"; do
        for inputType in "${inputTypes[@]}"; do
        # Execute the command with the current size and thread count
            sbatch bubble.grace_job $threads $size $inputType
        done
    done
done