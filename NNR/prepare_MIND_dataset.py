import collections
import csv
import os
import json
import shutil
import random
import numpy as np


# 재현성 위해
random.seed(0)
np.random.seed(0)

# 프로젝트 루트(예: NNR 폴더 상위) 및 데이터셋 경로 설정
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')

# MIND-small raw 데이터 경로
MIND_SMALL_TRAIN_DIR = os.path.join(DATASET_DIR, 'mind_small_train')
MIND_SMALL_DEV_DIR   = os.path.join(DATASET_DIR, 'mind_small_dev')

# MIND-200k raw 데이터 경로
MIND_200K_TRAIN_DIR = os.path.join(DATASET_DIR, 'mind_200k_train')
MIND_200K_DEV_DIR   = os.path.join(DATASET_DIR, 'mind_200k_dev')
MIND_200K_TEST_DIR  = os.path.join(DATASET_DIR, 'mind_200k_test')

# MIND-large raw 데이터 경로
MIND_LARGE_TRAIN_DIR = os.path.join(DATASET_DIR, 'mind_large_train')
MIND_LARGE_DEV_DIR   = os.path.join(DATASET_DIR, 'mind_large_dev')
MIND_LARGE_TEST_DIR  = os.path.join(DATASET_DIR, 'mind_large_test')

# 소규모 전처리 결과를 저장할 폴더 (예: mind_small_preprocessed/{train,dev,test})
MIND_SMALL_PREPROCESSED_DIR = os.path.join(DATASET_DIR, 'mind_small_preprocessed')
os.makedirs(MIND_SMALL_PREPROCESSED_DIR, exist_ok=True)
PREPROCESSED_TRAIN_DIR = os.path.join(MIND_SMALL_PREPROCESSED_DIR, 'train')
PREPROCESSED_DEV_DIR   = os.path.join(MIND_SMALL_PREPROCESSED_DIR, 'dev')
PREPROCESSED_TEST_DIR  = os.path.join(MIND_SMALL_PREPROCESSED_DIR, 'test')


def split_training_behaviors():
    """
    MIND_small_train/behaviors.tsv 파일을 95:5 비율로 섞은 후 train과 dev로 분리합니다.
    """
    ratio = 0.95
    behaviors_file = os.path.join(MIND_SMALL_TRAIN_DIR, 'behaviors.tsv')
    with open(behaviors_file, 'r', encoding='utf-8') as f:
        lines = [line for line in f if line.strip()]
    
    random.shuffle(lines)
    split_index = int(len(lines) * ratio)
    train_lines = lines[:split_index]
    dev_lines = lines[split_index:]
    return train_lines, dev_lines


def prepare_MIND_small():
    """
    소규모(MIND_small) 데이터셋 전처리:
      - raw behaviors.tsv (MIND_SMALL_TRAIN_DIR)를 95:5로 분할하여 train과 dev 세트를 생성
      - news.tsv는 MIND_SMALL_TRAIN_DIR의 파일을 train과 dev에 복사
      - 테스트 세트는 MIND_SMALL_DEV_DIR의 behaviors.tsv와 news.tsv를 그대로 복사
      - 결과는 MIND_SMALL_PREPROCESSED_DIR/{train,dev,test}에 저장
    """
    train_lines, dev_lines = split_training_behaviors()
    
    # 출력 폴더 생성
    os.makedirs(PREPROCESSED_TRAIN_DIR, exist_ok=True)
    os.makedirs(PREPROCESSED_DEV_DIR, exist_ok=True)
    os.makedirs(PREPROCESSED_TEST_DIR, exist_ok=True)
    
    # Train 세트 저장
    with open(os.path.join(PREPROCESSED_TRAIN_DIR, 'behaviors.tsv'), 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    src_news_train = os.path.join(MIND_SMALL_TRAIN_DIR, 'news.tsv')
    dst_news_train = os.path.join(PREPROCESSED_TRAIN_DIR, 'news.tsv')
    if os.path.exists(src_news_train):
        shutil.copy(src_news_train, dst_news_train)
    
    # Dev 세트 저장
    with open(os.path.join(PREPROCESSED_DEV_DIR, 'behaviors.tsv'), 'w', encoding='utf-8') as f:
        f.writelines(dev_lines)
    dst_news_dev = os.path.join(PREPROCESSED_DEV_DIR, 'news.tsv')
    if os.path.exists(src_news_train):
        shutil.copy(src_news_train, dst_news_dev)
    
    # Test 세트 저장 (소규모의 경우 dev 폴더의 파일 사용)
    src_behaviors_test = os.path.join(MIND_SMALL_DEV_DIR, 'behaviors.tsv')
    src_news_test = os.path.join(MIND_SMALL_DEV_DIR, 'news.tsv')
    dst_behaviors_test = os.path.join(PREPROCESSED_TEST_DIR, 'behaviors.tsv')
    dst_news_test = os.path.join(PREPROCESSED_TEST_DIR, 'news.tsv')
    if os.path.exists(src_behaviors_test):
        shutil.copy(src_behaviors_test, dst_behaviors_test)
    if os.path.exists(src_news_test):
        shutil.copy(src_news_test, dst_news_test)
    
    print(MIND_SMALL_PREPROCESSED_DIR)


def generate_knowledge_entity_embedding(data_mode):
    """
    각 데이터셋(MIND_small 또는 MIND_large)에 대해,
    raw 폴더에 있는 entity_embedding.vec 파일을 복사하여
    전처리된 폴더에 저장합니다.
    
    [소규모(MIND_small)]
      - train: MIND_SMALL_TRAIN_DIR -> PREPROCESSED_TRAIN_DIR
      - dev:   MIND_SMALL_DEV_DIR   -> PREPROCESSED_DEV_DIR
      - test:  MIND_SMALL_DEV_DIR   -> PREPROCESSED_TEST_DIR (소규모는 test에 dev 데이터를 사용)
    """
    
    if data_mode == 'small':
        dirs = {
            'train': (MIND_SMALL_TRAIN_DIR, PREPROCESSED_TRAIN_DIR),
            'dev':   (MIND_SMALL_DEV_DIR, PREPROCESSED_DEV_DIR),
            'test':  (MIND_SMALL_DEV_DIR, PREPROCESSED_TEST_DIR)
        }
    else:
        raise ValueError("data_mode는 'small'이어야 합니다.")
    
    for mode, (src_dir, target_dir) in dirs.items():
        os.makedirs(target_dir, exist_ok=True)
        src_file = os.path.join(src_dir, 'entity_embedding.vec')
        target_file = os.path.join(target_dir, 'entity_embedding.vec')
        if os.path.exists(src_file):
            shutil.copy(src_file, target_file)
        else:
            print(f"[{mode}] Warning: 소스 파일 {src_file}이 존재하지 않습니다.")
        
        src_file = os.path.join(src_dir, 'relation_embedding.vec')
        target_file = os.path.join(target_dir, 'relation_embedding.vec')
        if os.path.exists(src_file):
            shutil.copy(src_file, target_file)
        else:
            print(f"[{mode}] Warning: 소스 파일 {src_file}이 존재하지 않습니다.")


def prepare_MIND_large():
    """
    MIND_large 데이터셋 준비:
      - mind_large_train, mind_large_dev, mind_large_test 폴더에 behaviors.tsv, news.tsv, entity_embedding.vec 파일이 있는지 확인.
    """
    for d in [MIND_LARGE_TRAIN_DIR, MIND_LARGE_DEV_DIR, MIND_LARGE_TEST_DIR]:
        for fname in ['behaviors.tsv', 'news.tsv', 'entity_embedding.vec']:
            fpath = os.path.join(d, fname)
            if not os.path.exists(fpath):
                print(f"Error: {fpath} not found.")


def prepare_MIND_200k():
    """
    MIND_200k 데이터셋 준비:
      - mind_large_train, mind_large_dev, mind_large_test 폴더에 behaviors.tsv, news.tsv, entity_embedding.vec 파일이 있는지 확인.
    """
    for d in [MIND_200K_TRAIN_DIR, MIND_200K_DEV_DIR, MIND_200K_TEST_DIR]:
        for fname in ['behaviors.tsv', 'news.tsv', 'entity_embedding.vec']:
            fpath = os.path.join(d, fname)
            if not os.path.exists(fpath):
                print(f"Error: {fpath} not found.")


def merge_files(file_list, out_file, encoding="utf-8"):
    """
    주어진 file_list의 텍스트 파일들을 읽어 중복된 뉴스 id가 있을 경우 건너뛰고,
    모두 합친 후 out_file에 저장합니다.
    
    Args:
        file_list (list of str): 합칠 파일 경로 리스트.
        out_file (str): 저장할 출력 파일 경로.
        encoding (str): 파일 인코딩 (기본: utf-8).
    """
    merged_lines = []
    seen_ids = set()
    
    for file in file_list:
        if os.path.exists(file):
            with open(file, "r", encoding=encoding) as f:
                for line in f:
                    if not line.strip():
                        continue  # 빈 줄 건너뜀
                    # 첫번째 토큰이 ID
                    id = line.strip().split()[0]
                    if id in seen_ids:
                        continue
                    seen_ids.add(id)
                    merged_lines.append(line)
        else:
            print(f"Warning: {file} 파일이 존재하지 않습니다.")

    # 출력 파일에 합쳐진 내용을 저장
    with open(out_file, "w", encoding=encoding) as f_out:
        f_out.writelines(merged_lines)
    
    print(f">> {out_file} 파일로 저장 완료. (총 {len(merged_lines)} lines)")


def prepare_merged_dataset(data_mode):
    """
    mind_small 또는 mind_large 데이터셋에 대해, train, dev, test 세트를 합쳐
    하나의 merged 폴더를 생성합니다.
    
    Args:
        data_mode (str): "small" 또는 "large"
    """
    if data_mode == 'small':
        source_dirs = [
            PREPROCESSED_TRAIN_DIR,
            PREPROCESSED_DEV_DIR,
            PREPROCESSED_TEST_DIR
        ]
        merged_dir = os.path.join(DATASET_DIR, "mind_small_merged")
    elif data_mode == '200k':
        source_dirs = [
            MIND_200K_TRAIN_DIR,
            MIND_200K_DEV_DIR,
            MIND_200K_TEST_DIR
        ]
        merged_dir = os.path.join(DATASET_DIR, "mind_200k_merged")
    elif data_mode == 'large':
        source_dirs = [
            MIND_LARGE_TRAIN_DIR,
            MIND_LARGE_DEV_DIR,
            MIND_LARGE_TEST_DIR
        ]
        merged_dir = os.path.join(DATASET_DIR, "mind_large_merged")
    else:
        raise ValueError("data_mode는 'small' 또는 'large'여야 합니다.")
    
    # merged_dir 생성 (이미 존재하면 삭제 후 재생성)
    if os.path.exists(merged_dir):
        shutil.rmtree(merged_dir)
    os.makedirs(merged_dir, exist_ok=True)
    print(f">> Merged folder 생성: {merged_dir}")
    
    # 합칠 파일 목록: news.tsv, entity_embedding.vec
    filenames = ["news.tsv", "entity_embedding.vec", "relation_embedding.vec"]
    
    for fname in filenames:
        file_list = [os.path.join(src, fname) for src in source_dirs]
        out_file = os.path.join(merged_dir, fname)
        merge_files(file_list, out_file)
    
    print(">> 모든 세트 병합 완료.")


if __name__ == '__main__':
    prepare_MIND_small()
    generate_knowledge_entity_embedding('small')
    prepare_merged_dataset("small")
    print("==== MIND_small preparation complete ====")
    
    prepare_MIND_200k()
    prepare_merged_dataset("200k")
    print("==== MIND_200k preparation complete ====")

    prepare_MIND_large()
    prepare_merged_dataset("large")
    print("==== MIND_large preparation complete ====")
