#!/bin/bash
# extract_preprocess_MIND.sh

# dataset 폴더가 없으면 생성
mkdir -p dataset
cd dataset

########################################
# MINDlarge 압축 해제
########################################
if [ ! -d "mind_large_train" ]; then
    echo "Extracting MINDlarge_train.zip into mind_large_train..."
    unzip MINDlarge_train.zip -d mind_large_train
else
    echo "Directory mind_large_train exists, skipping extraction."
fi

if [ ! -d "mind_large_dev" ]; then
    echo "Extracting MINDlarge_dev.zip into mind_large_dev..."
    unzip MINDlarge_dev.zip -d mind_large_dev
else
    echo "Directory mind_large_dev exists, skipping extraction."
fi

if [ ! -d "mind_large_test" ]; then
    echo "Extracting MINDlarge_test.zip into mind_large_test..."
    unzip MINDlarge_test.zip -d mind_large_test
else
    echo "Directory mind_large_test exists, skipping extraction."
fi

########################################
# MINDsmall 압축 해제
########################################
if [ ! -d "mind_small_train" ]; then
    echo "Extracting MINDsmall_train.zip into mind_small_train..."
    unzip MINDsmall_train.zip -d mind_small_train
else
    echo "Directory mind_small_train exists, skipping extraction."
fi

if [ ! -d "mind_small_dev" ]; then
    echo "Extracting MINDsmall_dev.zip into mind_small_dev..."
    unzip MINDsmall_dev.zip -d mind_small_dev
else
    echo "Directory mind_small_dev exists, skipping extraction."
fi

########################################
# WikiKG90Mv2 압축 해제
########################################
if [ ! -d "wikikg90mv2" ]; then
    echo "Extracting wikikg90mv2_mapping.zip into wikikg90mv2..."
    unzip wikikg90mv2_mapping.zip -d wikikg90mv2
    mv ./wikikg90mv2/wikikg90mv2_mapping/* ./wikikg90mv2
    rm -rf ./wikikg90mv2/wikikg90mv2_mapping
else
    echo "Directory wikikg90mv2 exists, skipping extraction."
fi

########################################
# Python 전처리 스크립트 실행
########################################
cd ..
python NNR/prepare_MIND_dataset.py
