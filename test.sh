#!/bin/bash

ENV_NAME="LKPNR_CUDA"

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

cd ./NNR

python -u main.py --mode=test --test_model_path=./best_model/small/MHSA-MHSA/run_1/MHSA-MHSA.pt --news_encoder=MHSA --user_encoder=MHSA
