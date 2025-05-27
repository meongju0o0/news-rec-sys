import argparse
import pandas as pd
import json
import pickle


def extract_wikidata_ids(entity_str, source, news_id):
    """JSON 문자열에서 'Q'로 시작하는 Wikidata ID 추출"""
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


def create_link_entity_dic(tsv_file, output_pkl):
    """
    news.tsv 파일을 읽고, 뉴스 ID와 연결된 지식 엔터티 Wikidata ID를 추출하여 딕셔너리를 생성하고 저장
    """
    print("=" * 40)
    print("Reading TSV with pandas:", tsv_file)

    df = pd.read_csv(tsv_file, sep='\t', header=None, quoting=3, dtype=str,
                     names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'])

    link_entity_dic = {}

    for _, row in df.iterrows():
        news_id = row['news_id']
        if not isinstance(news_id, str) or not news_id.startswith("N"):
            continue

        title_entities_str = row['title_entities']
        abstract_entities_str = row['abstract_entities']

        wikidata_ids = set()
        wikidata_ids.update(extract_wikidata_ids(title_entities_str, "title_entities", news_id))
        wikidata_ids.update(extract_wikidata_ids(abstract_entities_str, "abstract_entities", news_id))

        link_entity_dic[news_id] = list(wikidata_ids)

    print("Finished processing. Sample output:")
    for i, (k, v) in enumerate(link_entity_dic.items()):
        print(f"{k}: {v}")
        if i >= 9: break  # Show first 10

    with open(output_pkl, 'wb') as f:
        pickle.dump(link_entity_dic, f)
    print(f"\nlink_entity_dic saved to {output_pkl}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create link entity dictionary from TSV files using pandas")
    parser.add_argument('--dataset', type=str, default='small', choices=['small', 'large', '200k'],
                        help='Dataset type: small, large, or 200k')
    args = parser.parse_args()

    tsv_file = f'../dataset/mind_{args.dataset}_merged/news.tsv'
    output_pkl = './link_entity_dic.pkl'

    create_link_entity_dic(tsv_file, output_pkl)
