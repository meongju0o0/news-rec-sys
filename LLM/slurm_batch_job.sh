#!/usr/bin/bash

#SBATCH -J LKPNR_LLM
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ce_ugrad
#SBATCH -w moana-y6
#SBATCH -t 1-0
#SBATCH -o ./logs/slurm-%A.out

pwd

ENV_NAME="CHAT_GLM_2"
eval "$(conda shell.bash hook)"

if ! conda info --envs | grep -q "$ENV_NAME"; then
  echo "There is no valid environment"
  exit 1
fi

conda activate "$ENV_NAME"

which python

python -u get_item.py --dataset="small" && \
python -u get_item.py --dataset="large"

exit 0
