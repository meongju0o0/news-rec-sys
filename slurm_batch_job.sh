#!/usr/bin/bash

#SBATCH -J LKPNR
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ce_ugrad
#SBATCH -w moana-y1
#SBATCH -t 1-0
#SBATCH -o ./logs/slurm-%A.out

echo "Current working directory: $(pwd)"
echo "Starting job on $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Node name: $SLURM_NODELIST"
echo "Number of nodes: $SLURM_NNODES"
echo "Number of GPUs: $SLURM_GPUS_ON_NODE"
echo "Number of CPUs: $SLURM_CPUS_ON_NODE"
echo "Memory per GPU: $SLURM_MEM_PER_GPU"
echo "Total memory: $SLURM_MEM_PER_NODE"

./train.sh

exit 0
