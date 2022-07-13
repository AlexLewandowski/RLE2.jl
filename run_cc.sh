#!/bin/bash
#SBATCH --account=def-schuurma
#SBATCH --time=3:00:00
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=32
#SBATCH --mem=0


julia experiment/run_parallel.jl --config_path $1 --num_workers 320
