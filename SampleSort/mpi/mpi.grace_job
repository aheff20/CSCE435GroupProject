#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE            #Do not propagate environment
#SBATCH --get-user-env=L         #Replicate login environment
#
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=JobName       #Set the job name to "JobName"
#SBATCH --time=00:03:00           #Set the wall clock limit
#SBATCH --nodes=1               #Request nodes
#SBATCH --ntasks-per-node=8
#SBATCH --mem=8G                 #Request GB per node 
#SBATCH --output=outs/reverse/output.%j       #Send stdout/err to "output.[jobID]" 
#
##OPTIONAL JOB SPECIFICATIONS
##SBATCH --mail-type=ALL              #Send email on all job events
##SBATCH --mail-user=email_address    #Send all emails to email_address 
#
##First Executable Line
#
array_size=$1
processes=$2
input_type=$3

module load intel/2020b       # load Intel software stack
module load CMake/3.12.1

CALI_CONFIG="spot(output=outs/reverse/p${processes}/sample_mpi_p${processes}-a${array_size}_${input_type}.cali, \
    time.variance)" \
mpirun -np $processes ./sample_mpi $array_size $input_type