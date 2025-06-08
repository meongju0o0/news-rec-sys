#!/bin/bash

ENV_NAME="LKPNR_CPU"
dataset="200k"

eval "$(conda shell.bash hook)"

if ! conda info --envs | grep -q "$ENV_NAME"; then
  echo "There is no valid environment"
  exit 1
fi

conda activate "$ENV_NAME"

rm -rf ./output.log

cd ./NNR

nohup python -u MIND_entity_subgraphs.py --dataset=$dataset 2>&1 | tee ../output.log &
