#!/bin/bash

# Array of input sizes (2^16, 2^18, 2^15 - 2^21 ..., 2^28) 65536 262144 1048576 4194304 16777216 67108864 268435456
inputSizes=( 32768 65536 131072 262144 524288 1048576  2097152 4194304  16777216 16777216 67108864  268435456)

# Array of CUDA threads (64, 128, ..., 4096) 64 128 256 512    1024 's' 'r' 'a' 'p' 
cudaThreads=( 1 64 128 256 512 1024 )
methods=( 'a' )

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
