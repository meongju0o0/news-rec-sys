#!/bin/bash

ENV_NAME="LKPNR_CUDA"
dataset="small"

eval "$(conda shell.bash hook)"

if ! conda info --envs | grep -q "$ENV_NAME"; then
  echo "There is no valid environment"
  exit 1
fi

conda activate "$ENV_NAME"

echo "Copy LLM News Embeddings from ./LLM/$dataset/item_emb.pkl to ./NNR"
cp ./LLM/$dataset/item_emb.pkl ./NNR

echo "Generate Graph for $dataset dataset"
cd ./graph
python -u preprocess_entity_emb.py --dataset=$dataset
python -u preprocess_news_entity.py --dataset=$dataset
cd ..

start_time=$(date +%s)

echo "Start Training for $dataset dataset"
cd ./NNR
python -u main.py --dataset=$dataset --news_encoder=MHSA --user_encoder=MHSA

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
hours=$(( elapsed / 3600 ))
minutes=$(( (elapsed % 3600) / 60 ))
seconds=$(( elapsed % 60 ))

echo "Total execution time: ${hours} hours, ${minutes} minutes, ${seconds} seconds."
echo "Training for $dataset dataset completed."