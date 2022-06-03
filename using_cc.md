# Using Compute Canada

Before submitting a job or starting an interactive session, do a `git pull` because you may not have internet access in the job/session.

Before submitting a job, and after doing a `git pull`, it is good practice to start an interactive session and compile the project. In Julia, do:
`using Pkg; Pkg.instantiate()`
This avoids wasting time on the requested cluster.

----------------------------------------------------------------------

(east coast server)
https://docs.alliancecan.ca/wiki/Graham
`ssh user@graham.computecanada.ca`

Before running gpu job with julia:
`module load julia/1.6.2 cuda cudnn`
`export CUDA_ROOT=$EBROOTCUDNN`

When you run julia on graham, specify not to download CUDA:

`JULIA_CUDA_USE_BINARYBUILDER=false julia`

----------------------------------------------------------------------
(west coast server)
https://docs.computecanada.ca/wiki/Cedar
`ssh user@cedar.computecanada.ca`

Before running gpu job with julia:
module load julia/1.6.2 cuda cudnn

----------------------------------------------------------------------
To run an interactive job:

`salloc --time=1:0:0 --cpus-per-task=2 --account=def-<acc> --mem-per-cpu=4000 --gres=gpu:1`

This requests a node for 1 hour with 2 cpus, 8gb ram and 1 gpu.

----------------------------------------------------------------------
To run a batch job, which submits a script:

On Graham:
`sbatch run_cc_gpu_graham.sh <path to config file>

On Cedar:
`sbatch run_cc_gpu_cedar.sh <path to config file>
----------------------------------------------------------------------

To check the status of your submitted job:

`squeue -u user`

----------------------------------------------------------------------
To check usage at the group level:

`sshare -l -A def-<acc>_gpu --all` (GPU)

`sshare -l -A def-<acc>_cpu --all` (CPU)

Lower LevelFS means you have less priority.
Note: if you did not submit a job sometimes RawShares will be 1 and LevelFS will be very low. 
This is not accurate, RawShares is something like 20k and it will update some time after you submit a job.
You can read more about Fair Share in the compute canada docs website





    
