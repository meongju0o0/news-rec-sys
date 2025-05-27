import argparse
import os
import pickle
import json
import pandas as pd


def load_embedding_file(file_path):
    """
    주어진 entity_embedding.vec 파일을 읽어, 노드와 임베딩 정보를 딕셔너리로 반환
    """
    emb_dict = {}
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if not parts:
                continue
            node = parts[0]
            try:
                embedding = [float(v) for v in parts[1:]]
            except Exception as e:
                print(f"Error parsing line in {file_path}: {line}\n{e}")
                continue
            if node not in emb_dict:
                emb_dict[node] = embedding
    print(f"Loaded {len(emb_dict)} embeddings from {file_path}")
    return emb_dict


def extract_wikidata_ids(entity_str, source, news_id):
    """
    JSON 문자열에서 'Q'로 시작하는 Wikidata ID 추출
    """
    wikidata_ids = []
    if pd.isna(entity_str) or entity_str.strip() == "":
        return wikidata_ids
    try:
        entities = json.loads(entity_str)
        for entity in entities:
            wikidata_id = entity.get("WikidataId")
            if wikidata_id and wikidata_id.startswith("Q"):
                wikidata_ids.append(wikidata_id)
            elif not wikidata_id:
                print(f"[Warning] Missing WikidataId in {source} for {news_id}: {entity}")
    except json.JSONDecodeError as e:
        print(f"[Error] JSON decode error in {source} for {news_id}: {e}")
    return wikidata_ids


def parse_entities_from_tsv(tsv_path):
    """
    TSV 파일에서 title_entities, abstract_entities에 포함된 Wikidata ID들을 추출하여 집합으로 반환
    """
    print('*' * 16, "Parse Wikidata ID", '*' * 16)

    df = pd.read_csv(tsv_path, sep='\t', header=None, quoting=3, dtype=str,
                     names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'])

    wikidata_ids = set()

    for _, row in df.iterrows():
        news_id = row['news_id']
        if not isinstance(news_id, str) or not news_id.startswith("N"):
            continue

        for field in ['title_entities', 'abstract_entities']:
            entity_str = row.get(field)
            wikidata_ids.update(extract_wikidata_ids(entity_str, field, news_id))
            
    print(f"Total unique Wikidata IDs: {len(wikidata_ids)}")
    for i, entity in enumerate(list(wikidata_ids)[:10]):
        print(f"Entity {i}: {entity}")
    print('*' * 16, "Parse Wikidata ID", '*' * 16)
    return wikidata_ids


def update_embedding_dict_with_missing_entities(embedding_dict, required_entities):
    """
    required_entities에 포함된 엔티티 중 embedding_dict에 없는 엔티티에 대해 zero vector 추가
    """
    assert embedding_dict, "Embedding dictionary is empty."

    emb_dim = len(next(iter(embedding_dict.values())))
    missing = 0
    existing = 0

    for entity in required_entities:
        if entity not in embedding_dict:
            embedding_dict[entity] = [0.0] * emb_dim
            missing += 1
        else:
            existing += 1

    print(f"총 추출된 엔티티: {len(required_entities)}")
    print(f"이미 임베딩 존재: {existing}")
    print(f"새로 zero vector 생성: {missing}")
    return embedding_dict


def save_embedding_dict(embedding_dict, output_pkl):
    with open(output_pkl, 'wb') as f:
        pickle.dump(embedding_dict, f)
    print(f"Embedding dictionary saved to: {output_pkl}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load entity embedding and supplement with missing abstract entities.")
    parser.add_argument('--dataset', type=str, default='small', choices=['small', 'large', '200k'],
                        help='Dataset type: small, large, or 200k')
    args = parser.parse_args()

    vec_file = f'../dataset/mind_{args.dataset}_merged/entity_embedding.vec'
    tsv_file = f'../dataset/mind_{args.dataset}_merged/news.tsv'
    output_pkl = 'all_entity_emb_dic.pkl'

    # 1. 임베딩 불러오기
    embedding_dict = load_embedding_file(vec_file)

    # 2. TSV로부터 필요한 Wikidata 엔티티 추출
    required_entities = parse_entities_from_tsv(tsv_file)

    # 3. 누락된 엔티티 zero vector로 추가
    final_embedding_dict = update_embedding_dict_with_missing_entities(embedding_dict, required_entities)

    # 4. 저장
    save_embedding_dict(final_embedding_dict, output_pkl)
