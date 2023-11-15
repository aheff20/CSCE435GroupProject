#!/bin/bash

# Array of input sizes (2^16, 2^18, ..., 2^28) 268435456
inputSizes=(65536 262144 1048576 4194304 16777216 67108864 )

# Array of proc  (64, 128, ..., 128 256 512 1024 )
procs=(2 4 8 16 32 64 128 )

methods=('s' 'r' 'a' 'p')

# Iterate over each input size
for size in "${inputSizes[@]}"; do
    # Iterate over each thread count
    for procs in "${procs[@]}"; do

        for method in "${methods[@]}"; do
        # Execute the command with the current size and thread count
            sbatch mpi_merge.grace_job $size $procs $method
        done
    done
done


