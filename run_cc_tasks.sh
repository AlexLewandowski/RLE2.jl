#!/bin/bash
#SBATCH --account=def-schuurma
#SBATCH --time=3:00:00
#SBATCH --ntasks=20               # number of MPI processes
#SBATCH --mem-per-cpu=12000M      # memory; default unit is megabytes
module load julia/1.6.2

julia experiment/run_parallel.jl --config_path $1 --num_workers 20
