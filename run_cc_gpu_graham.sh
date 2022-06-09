#!/bin/bash
#SBATCH --account=def-schuurma
#SBATCH --nodes=16                          # Number of nodes
#SBATCH --ntasks-per-node=1                # number of MPI processes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0-12:00
module load julia/1.6.2 cuda cudnn
export CUDA_ROOT=$EBROOTCUDNN

JULIA_CUDA_USE_BINARYBUILDER=false julia experiment/run_parallel.jl --config_path $1 --num_workers 16
