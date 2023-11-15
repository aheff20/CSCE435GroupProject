#!/bin/bash

# Array of input sizes (2^16, 2^18, ..., 2^28)
inputSizes=(65536 262144 1048576 4194304 16777216 67108864 268435456)

# Array of CUDA threads (64, 128, ..., 4096)4 128 256 512 1024
cudaThreads=(64 128 256 512 1024)
methods=('s' 'r' 'a' 'p')

# Iterate over each input size
for size in "${inputSizes[@]}"; do
    # Iterate over each thread count
    for threads in "${cudaThreads[@]}"; do
        # Iterate over each method
        for method in "${methods[@]}"; do
            # Execute the command with the current size, thread count, and method
            sbatch merge.grace_job $threads $size $method
        done
    done
done
