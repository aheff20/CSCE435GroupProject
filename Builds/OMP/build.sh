#!/bin/bash

module purge

module load intel/2020b
module load GCCcore/8.3.0
module load Clang/9.0.1
module load CMake/3.12.1

CXX=/sw/eb/sw/Clang/9.0.1-GCCcore-8.3.0/bin/clang++ \
cmake \
    -Dcaliper_DIR=/scratch/group/csce435-f23/Caliper/caliper/share/cmake/caliper \
    -Dadiak_DIR=/scratch/group/csce435-f23/Adiak/adiak/lib/cmake/adiak \
    .

make