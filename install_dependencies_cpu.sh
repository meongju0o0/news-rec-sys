#!/bin/bash

# Python 3.11 + Torch 2.1.2 + CPU 환경 구성

ENV_NAME="LKPNR_CPU"

eval "$(conda shell.bash hook)"

if ! conda info --envs | grep -q "$ENV_NAME"; then
  echo "There is no valid environment"
  exit 1
fi

conda activate "$ENV_NAME"

TORCH_VERSION=2.1.2
TORCHTEXT_VERSION=0.16.2
CUDA_VERSION=cpu

pip install --upgrade pip
pip install ninja

# PyTorch 설치
pip install torch==${TORCH_VERSION} torchtext==${TORCHTEXT_VERSION} --index-url https://download.pytorch.org/whl/${CUDA_VERSION}
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+${CUDA_VERSION}.html
pip install torch_geometric -f https://data.pyg.org/whl/torch-2.1.0+${CUDA_VERSION}.html
pip install pyg-lib -f https://data.pyg.org/whl/torch-2.1.0+${CUDA_VERSION}.html
pip install torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.1.2+${CUDA_VERSION}.html

# 기타 라이브러리 설치
pip install 'numpy<2'
pip install -U scikit-learn
pip install dill ogb
pip install --user -U nltk
pip install nltk
pip install pcst_fast