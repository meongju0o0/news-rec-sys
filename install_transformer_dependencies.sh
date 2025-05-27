#!/bin/bash

# the versions of torch and torchtext must be matched (https://pypi.org/project/torchtext)
# the CUDA version must be matched with torch-scatter (https://github.com/rusty1s/pytorch_scatter)


ENV_NAME="CHAT_GLM_2"

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

TORCH_VERSION=2.6.0
CUDA_VERSION=cu124

pip3 install torch==${TORCH_VERSION} torchvision torchaudio --index-url https://download.pytorch.org/whl/${CUDA_VERSION}
pip install protobuf transformers==4.30.2 cpm_kernels gradio mdtex2html sentencepiece accelerate
pip install nltk
