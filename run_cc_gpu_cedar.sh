#!/bin/bash
#SBATCH --account=def-schuurma
#SBATCH --ntasks=16                # number of MPI processes
#SBATCH --nodes=16                 # Number of nodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=0-12:00
module load julia/1.6.2 cuda cudnn

julia experiment/run_parallel.jl --config_path $1 --num_workers 16
