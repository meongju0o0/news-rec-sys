import argparse
import os
import pickle
import json
import csv

import torch

import pandas as pd
import numpy as np

from collections import defaultdict
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import subgraph
from ogb.lsc import WikiKG90Mv2Dataset
from pcst_fast import pcst_fast
from tqdm import tqdm


def load_wikikg_map():
    df = pd.read_csv(
        '../dataset/wikikg90mv2_mapping/entity.csv',
        usecols=['idx', 'entity'],
        dtype={'idx': int, 'entity': str},
        encoding='utf-8',
        engine='c',
        quoting=csv.QUOTE_MINIMAL,
    )
    ent2idx = dict(zip(df['entity'], df['idx']))
    idx2ent = dict(zip(df['idx'], df['entity']))
    return ent2idx, idx2ent


def load_wikikg_rel_map():
    df = pd.read_csv(
        '../dataset/wikikg90mv2_mapping/relation.csv',
        usecols=['idx', 'relation'],
        dtype={'idx': int, 'relation': str},
        encoding='utf-8',
        engine='c',
        quoting=csv.QUOTE_MINIMAL,
    )
    rel2idx = dict(zip(df['relation'], df['idx']))
    idx2rel = dict(zip(df['idx'], df['relation']))
    return rel2idx, idx2rel


def load_linked_entity_map():
    with open('../graph/link_entity_dic.pkl', 'rb') as f:
        link_entity_dic = pickle.load(f)
    return link_entity_dic


def extract_wikidata_ids(entity_str, source, news_id):
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
    df = pd.read_csv(tsv_path, sep='\t', header=None, quoting=3, dtype=str,
                     names=['news_id','cat','subcat','title','abs','url','title_entities','abstract_entities'])
    ids = set()
    for _, row in df.iterrows():
        news_id = row['news_id']
        for field in ['title_entities', 'abstract_entities']:
            entity_str = row.get(field, "")
            ids.update(extract_wikidata_ids(entity_str, field, news_id))
    return list(ids)


def parse_entity_emb_vec(entity_emb_path):
    entity_feat = {}
    with open(entity_emb_path, 'r', encoding='utf-8') as f:
        for line in f:
            if len(line.strip()) > 0:
                terms = line.strip().split('\t')
                assert len(terms) == 101, 'entity embedding dim does not match'
                wikidata_id = terms[0]
                if wikidata_id not in entity_feat:
                    entity_feat[wikidata_id] = np.array(terms[1:], dtype=np.float32)
    return entity_feat


def parse_rel_emb_vec(relation_emb_path):
    relation_feat = {}
    with open(relation_emb_path, 'r', encoding='utf-8') as f:
        for line in f:
            if len(line.strip()) > 0:
                terms = line.strip().split('\t')
                assert len(terms) == 101, 'relation embedding dim does not match'
                relation_id = terms[0]
                if relation_id not in relation_feat:
                    relation_feat[relation_id] = np.array(terms[1:], dtype=np.float32)
    return relation_feat


def generate_induced_subgraph(
    wikikg: WikiKG90Mv2Dataset,
    wikidata_ids: list[str],
    entity_feat: dict[str, np.ndarray],
    relation_feat: dict[str, np.ndarray],
    idx2ent: dict[int, str],
    ent2idx: dict[str, int],
    idx2rel: dict[int, str]
) -> Data:
    # 뉴스 엔티티 → seed_nodes
    idxs = np.fromiter((ent2idx[q] for q in wikidata_ids if q in ent2idx),
                       dtype=np.int64)
    seed_nodes = np.unique(idxs)

    # seed_nodes 포함 triple 필터링
    triples = wikikg.train_hrt
    idx = np.flatnonzero(
        np.in1d(triples[:, 0], seed_nodes, assume_unique=True) &
        np.in1d(triples[:, 2], seed_nodes, assume_unique=True)
    )
    print(f"Found {len(idx)} triples containing seed nodes.")
    sub_triples = triples[idx]

    # 완전 중복 (h,r,t) 제거
    _, uniq_idx = np.unique(sub_triples, axis=0, return_index=True)
    sub_triples = sub_triples[uniq_idx]

    # wikikg edge index 매핑
    original_edge_idx = idx[uniq_idx].astype(np.int64)

    # 등장한 노드 집합
    heads = sub_triples[:, 0]
    rels = sub_triples[:, 1]
    tails = sub_triples[:, 2]
    all_nodes = np.unique(np.concatenate((heads, tails, seed_nodes)))
    print(f"Total unique nodes in subgraph: {all_nodes.size}")

    # local-index 매핑
    heads_loc = np.searchsorted(all_nodes, heads)
    tails_loc = np.searchsorted(all_nodes, tails)

    # 노드 피처: MIND 임베딩으로 대체 (default: 0, missing은 이후 보정)
    default_emb = np.zeros((100,), dtype=np.float32)
    qids = [idx2ent[nid] for nid in all_nodes]
    node_feats = np.stack(
        [entity_feat.get(qid, default_emb) for qid in qids],
        axis=0
    )
    missing_nodes = int(np.sum(np.all(node_feats == 0, axis=1)))
    x_sub = torch.from_numpy(node_feats).float()

    # 엣지 피처: MIND 임베딩으로 대체 (default: 0, missing은 이후 보정)
    rel_strs = [idx2rel[rid] for rid in rels]
    edge_feats = np.stack(
        [relation_feat.get(rel_str, default_emb) for rel_str in rel_strs],
        axis=0
    )
    missing_edges = int(np.sum(np.all(edge_feats == 0, axis=1)))
    ea_sub = torch.from_numpy(edge_feats).float()

    print(f"Total nodes: {all_nodes.size}, Total edges: {len(original_edge_idx)}")
    print(f"Missing nodes: {missing_nodes}, Missing edges: {missing_edges}")

    # Missing node embedding 보정: 1-hop 이웃 평균
    neighbors = defaultdict(list)
    for h_l, t_l in zip(heads_loc, tails_loc):
        neighbors[h_l].append(t_l)
        neighbors[t_l].append(h_l)

    missing_nodes = 0
    for i in range(x_sub.shape[0]):
        if torch.all(x_sub[i] == 0):
            neighs = neighbors[i]
            valid_neighs = [x_sub[j] for j in neighs if not torch.all(x_sub[j] == 0)]
            if valid_neighs:
                x_sub[i] = torch.stack(valid_neighs).mean(dim=0)
            else:
                missing_nodes += 1

    # Missing edge embedding 보정: 연결된 노드 평균
    missing_edges = 0
    for i in range(ea_sub.shape[0]):
        if torch.all(ea_sub[i] == 0):
            h_l = heads_loc[i]
            t_l = tails_loc[i]
            ea_sub[i] = (x_sub[h_l] + x_sub[t_l]) / 2
            if torch.all(ea_sub[i] == 0):  # 여전히 0이면 missing
                missing_edges += 1

    print(f"Total nodes: {all_nodes.size}, Total edges: {len(original_edge_idx)}")
    print(f"Remaining missing nodes (no neighbor info): {missing_nodes}")
    print(f"Remaining missing edges (nodes also missing): {missing_edges}")

    # 엣지 인덱스
    ei_sub = torch.from_numpy(np.vstack((heads_loc, tails_loc)))

    # Data 객체 생성
    data = Data(
        x=x_sub,
        edge_index=ei_sub,
        edge_attr=ea_sub,
        num_nodes=all_nodes.size
    )
    data.global_node_idx = all_nodes
    data.global_edge_idx = original_edge_idx

    return data


def retrieval_via_pcst(graph, q_emb, root, topk=3, topk_e=3, cost_e=0.5, pruning='strong'):
    c = 0.01

    num_clusters = 1
    verbosity_level = 0
    if topk > 0:
        n_prizes = torch.nn.CosineSimilarity(dim=-1)(q_emb, graph.x)
        topk = min(topk, graph.num_nodes)
        _, topk_n_indices = torch.topk(n_prizes, topk, largest=True)

        n_prizes = torch.zeros_like(n_prizes)
        n_prizes[topk_n_indices] = torch.arange(topk, 0, -1, dtype=torch.float32)
    else:
        n_prizes = torch.zeros(graph.num_nodes)

    if topk_e > 0:
        e_prizes = torch.nn.CosineSimilarity(dim=-1)(q_emb, graph.edge_attr)
        topk_e = min(topk_e, e_prizes.unique().size(0))

        topk_e_values, _ = torch.topk(e_prizes.unique(), topk_e, largest=True)
        e_prizes[e_prizes < topk_e_values[-1]] = 0.0
        last_topk_e_value = topk_e
        for k in range(topk_e):
            indices = e_prizes == topk_e_values[k]
            value = min((topk_e-k)/sum(indices), last_topk_e_value)
            e_prizes[indices] = value
            last_topk_e_value = value*(1-c)
        # reduce the cost of the edges such that at least one edge is selected
        cost_e = min(cost_e, e_prizes.max().item()*(1-c/2))
    else:
        e_prizes = torch.zeros(graph.edge_index.size(1))

    costs = []
    edges = []
    virtual_n_prizes = []
    virtual_edges = []
    virtual_costs = []
    mapping_n = {}
    mapping_e = {}
    for i, (src, dst) in enumerate(graph.edge_index.T.numpy()):
        prize_e = e_prizes[i]
        if prize_e <= cost_e:
            mapping_e[len(edges)] = i
            edges.append((src, dst))
            costs.append(cost_e - prize_e)
        else:
            virtual_node_id = graph.num_nodes + len(virtual_n_prizes)
            mapping_n[virtual_node_id] = i
            virtual_edges.append((src, virtual_node_id))
            virtual_edges.append((virtual_node_id, dst))
            virtual_costs.append(0)
            virtual_costs.append(0)
            virtual_n_prizes.append(prize_e - cost_e)

    prizes = np.concatenate([n_prizes, np.array(virtual_n_prizes)])
    num_edges = len(edges)
    if len(virtual_costs) > 0:
        costs = np.array(costs+virtual_costs)
        edges = np.array(edges+virtual_edges)

    vertices, edges = pcst_fast(edges, prizes, costs, root, num_clusters, pruning, verbosity_level)

    selected_nodes = vertices[vertices < graph.num_nodes]
    selected_edges = [mapping_e[e] for e in edges if e < num_edges]
    virtual_vertices = vertices[vertices >= graph.num_nodes]
    if len(virtual_vertices) > 0:
        virtual_vertices = vertices[vertices >= graph.num_nodes]
        virtual_edges = [mapping_n[i] for i in virtual_vertices]
        selected_edges = np.array(selected_edges+virtual_edges)

    edge_index = graph.edge_index[:, selected_edges]
    selected_nodes = np.unique(np.concatenate([selected_nodes, edge_index[0].numpy(), edge_index[1].numpy()]))

    mapping = {n: i for i, n in enumerate(selected_nodes.tolist())}

    x = graph.x[selected_nodes]
    edge_attr = graph.edge_attr[selected_edges]
    src = [mapping[i] for i in edge_index[0].tolist()]
    dst = [mapping[i] for i in edge_index[1].tolist()]
    edge_index = torch.LongTensor([src, dst])

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=len(selected_nodes)
    )

    data.global_node_idx = graph.global_node_idx[selected_nodes]
    data.global_edge_idx = graph.global_edge_idx[selected_edges]

    return data


def generate_pcst_subgraphs(
    induced_subg: Data,
    entity_feats: dict[str, np.ndarray],
    wikidata_ids: list[str],
    ent2idx: dict[str,int],
    idx2ent: dict[int,str],
    global2local: dict[int,int],
) -> dict[str, Data]:
    results = {}
    
    for wid in tqdm(wikidata_ids):
        if wid in results:
            continue
        idx = ent2idx.get(wid)
        if idx is None: 
            results[wid] = None
            continue

        # PCST 파라미터 설정
        topk, topk_e, cost_e, pruning = 3, 3, 0.5, 'gw'
        entity_feat = entity_feats.get(wid, None)
        if entity_feat is not None:
            q_emb = torch.from_numpy(np.array(entity_feats[wid], dtype=np.float32))
            root = global2local.get(idx)
            if root is None:
                results[wid] = None
                continue

            # PCST 서브그래프 생성
            pcst_subg = retrieval_via_pcst(induced_subg, q_emb, root, topk, topk_e, cost_e, pruning)

            # 딕셔너리에 캐싱
            for nid in pcst_subg.global_node_idx:
                wid = idx2ent.get(nid)
                if wid is None:
                    continue
                if wid not in results:
                    results[wid] = pcst_subg
                elif results.get(wid) is not None and results[wid].num_nodes < pcst_subg.num_nodes:
                    results[wid] = pcst_subg
        else:
            results[wid] = None
            
    return results


def sample_pcst_subgraphs(
    pcst_subg_dict: dict[str, Data],
    ent2idx: dict[str, int],
    num_neighbors: list[int]
) -> dict[str, Data]:
    updated_dict = {}
    not_found = []

    for wid, pcst_subg in tqdm(pcst_subg_dict.items()):
        if pcst_subg is None or wid not in ent2idx:
            updated_dict[wid] = None
            continue
        if pcst_subg.num_nodes < num_neighbors[0] + num_neighbors[1] + 1:
            updated_dict[wid] = pcst_subg
            continue

        global_node_idx = pcst_subg.global_node_idx
        root_global_idx = ent2idx[wid]

        mask = (global_node_idx == root_global_idx)
        nonzero_indices = np.where(mask)[0]
        if nonzero_indices.size == 0:
            not_found.append((wid, root_global_idx))
            updated_dict[wid] = None
            continue
        local_root = int(nonzero_indices[0])

        try:
            loader = NeighborLoader(
                data=pcst_subg,
                input_nodes=torch.tensor([local_root]),
                num_neighbors=num_neighbors,
                batch_size=1,
                shuffle=False,
                num_workers=0
            )
            batch = next(iter(loader))

            sub_data = Data(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                num_nodes=batch.num_nodes
            )

            n_id = batch.n_id.numpy() if isinstance(batch.n_id, torch.Tensor) else np.array(batch.n_id)
            e_id = batch.e_id.numpy() if isinstance(batch.e_id, torch.Tensor) else np.array(batch.e_id)

            sub_data.global_node_idx = pcst_subg.global_node_idx[n_id]
            sub_data.global_edge_idx = pcst_subg.global_edge_idx[e_id]

            updated_dict[wid] = sub_data

        except Exception as e:
            print(f"[Error] Sampling for {wid} failed: {e}")
            updated_dict[wid] = None

    for wid, root_gid in not_found:
        print(f"[Warning] root_global_idx {root_gid} not found in global_node_idx for wid={wid}")

    return updated_dict


def generate_news_subgraphs(
    link_entity_dic: dict[str, list[str]],
    dataset: str,
    induced_subg: Data,
    entity_subg_dict: dict[str, Data],
    global2local: dict[int, int],
    entity_dim: int = 100
):
    # prepare one dummy graph to reuse
    dummy_graph = Data(
        x=torch.zeros((1, entity_dim), dtype=induced_subg.x.dtype),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        edge_attr=torch.zeros((0, entity_dim), dtype=induced_subg.edge_attr.dtype),
        num_nodes=1
    )
    dummy_mask = torch.zeros(1, dtype=torch.bool)

    with open('news_ID-%s.json' % dataset, 'r', encoding='utf-8') as news_ID_f:
        news_ID_dict = json.load(news_ID_f)

    cnt_pad = 0
    cnt_no_entity_in_news = 0
    cnt_no_qid = 0
    cnt_no_global_id = 0
    news_subgraph = {}
    for news_id, news_idx in news_ID_dict.items():
        if news_id == '<PAD>':
            cnt_pad += 1
            news_subgraph[news_idx] = (dummy_graph, dummy_mask)
            continue

        qids = link_entity_dic.get(news_id, [])
        if not qids:
            cnt_no_entity_in_news += 1
            news_subgraph[news_idx] = (dummy_graph, dummy_mask)
            continue

        # collect all entity-global-IDs
        union_globals = set()
        for qid in qids:
            subg = entity_subg_dict.get(qid)
            if subg is not None:
                union_globals.update(subg.global_node_idx.tolist())

        if not union_globals:
            cnt_no_qid += 1
            news_subgraph[news_idx] = (dummy_graph, dummy_mask)
            continue

        # map to local indices
        subset = [global2local[g] for g in sorted(union_globals) if g in global2local]
        if len(subset) == 0:
            cnt_no_global_id += 1
            news_subgraph[news_idx] = (dummy_graph, dummy_mask)
            continue
        else:
            subset = torch.tensor(subset, dtype=torch.long)

        # extract induced subgraph
        edge_index_sub, edge_attr_sub, _ = subgraph(
            subset,
            induced_subg.edge_index,
            induced_subg.edge_attr,
            relabel_nodes=True,
            return_edge_mask=True
        )

        sub_data = Data(
            x=induced_subg.x[subset],
            edge_index=edge_index_sub,
            edge_attr=edge_attr_sub,
            num_nodes=subset.size(0)
        )

        # create seed mask
        local_idx_map = {g: i for i, g in enumerate(subset.tolist())}
        seed_mask = torch.zeros(subset.size(0), dtype=torch.bool)
        entity_ids = union_globals
        valid_ids = entity_ids & local_idx_map.keys()
        for g_id in valid_ids:
            seed_mask[local_idx_map[g_id]] = True

        news_subgraph[news_idx] = (sub_data, seed_mask)
    
    print(f"Total news IDs: {len(news_ID_dict)}")
    print(f" - <PAD> news: {cnt_pad}")
    print(f" - No entities in news: {cnt_no_entity_in_news}")
    print(f" - No entities in subgraph: {cnt_no_qid}")
    print(f" - No mapping to local indices: {cnt_no_global_id}")
    print(f"Total generated news subgraphs: {len(news_subgraph)}")

    return news_subgraph


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='small', choices=['small','large','200k'])
    args = parser.parse_args()

    # load
    print(f"Loading MIND-{args.dataset} & WikiKG90Mv2 dataset...")
    wikikg              = WikiKG90Mv2Dataset(root='../dataset')
    ent2idx, idx2ent    = load_wikikg_map()
    rel2idx, idx2rel    = load_wikikg_rel_map()
    wikidata_ids        = parse_entities_from_tsv(f'../dataset/mind_{args.dataset}_merged/news.tsv')
    linked_entities     = load_linked_entity_map()
    entity_embeddings   = parse_entity_emb_vec(f'../dataset/mind_{args.dataset}_merged/entity_embedding.vec')
    rel_embeddings      = parse_rel_emb_vec(f'../dataset/mind_{args.dataset}_merged/relation_embedding.vec')

    if os.path.exists(f'entity_mapping-{args.dataset}.pkl') and os.path.exists(f'induced_subgraph-{args.dataset}.pt'):
        print(f"induced_subgraph-{args.dataset}.pt already exists, loading...")
        with open(f'entity_mapping-{args.dataset}.pkl', 'rb') as f:
            mapping = pickle.load(f)
        local2global = mapping['local2global']
        global2local = mapping['global2local']
        induced_subg = torch.load(f'induced_subgraph-{args.dataset}.pt')
    else:
        print("Generating induced subgraph...")
        induced_subg = generate_induced_subgraph(wikikg, wikidata_ids, entity_embeddings, rel_embeddings, idx2ent, ent2idx, idx2rel)

        global_idx = induced_subg.global_node_idx.tolist()
        local2global = {local: global_ for local, global_ in enumerate(global_idx)}
        global2local = {global_: local for local, global_ in local2global.items()}

        mapping = {
            'local2global': local2global,
            'global2local': global2local,
        }
        with open(f'entity_mapping-{args.dataset}.pkl', 'wb') as f:
            pickle.dump(mapping, f)
        print(f"Saved index mapping to entity_mapping-{args.dataset}.pkl")

        save_path = f'induced_subgraph-{args.dataset}.pt'
        torch.save(induced_subg, save_path)
        print(f'induced_subg saved at: {save_path}')

    ent_ids = [ent2idx[wid] for wid in wikidata_ids if wid in ent2idx]
    induced_set = set(induced_subg.global_node_idx.tolist())
    included = [eid for eid in ent_ids if eid in induced_set]
    missing = [eid for eid in ent_ids if eid not in induced_set]

    print(f"Total entity ids in ent2idx: {len(ent_ids)}")
    print(f" - Found in induced_subgraph: {len(included)} ({len(included)/len(ent_ids)*100:.2f}%)")
    print(f" - Missing from induced_subgraph: {len(missing)} ({len(missing)/len(ent_ids)*100:.2f}%)")

    # Generate PCST subgraphs
    output_path = f"entity_pcst_subgraph-{args.dataset}.pkl"
    if os.path.exists(output_path):
        print(f"PCST subgraph dict already exists at entity_pcst_subgraph-{args.dataset}.pkl, loading...")
        with open(f'entity_pcst_subgraph-{args.dataset}.pkl', 'rb') as f:
            pcst_dict = pickle.load(f)
    else:
        # 각 QID별 PCST subgraph 생성
        print("Generating PCST subgraphs...")
        pcst_dict = generate_pcst_subgraphs(induced_subg, entity_embeddings, wikidata_ids, ent2idx, idx2ent, global2local)
        output_path = f"entity_pcst_subgraph-{args.dataset}.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(pcst_dict, f)
        print(f"PCST subgraph dict saved to {output_path}")
    print(f"Total entities: {len(wikidata_ids)}, Subgraphs generated: {sum(v is not None for v in pcst_dict.values())}")

    # Sample 2-hop subgraphs
    output_path = f"entity_2hop_subgraph-{args.dataset}.pkl"
    if os.path.exists(output_path):
        print(f"2-hop subgraph dict already exists at {output_path}, loading...")
        with open(output_path, 'rb') as f:
            sampled_subg_dict = pickle.load(f)
    else:
        # Sample subgraph
        print("Sampling subgraphs...")
        sampled_subg_dict = sample_pcst_subgraphs(pcst_dict, ent2idx, num_neighbors=[20, 10])
        # Save to pickle
        with open(output_path, 'wb') as f:
            pickle.dump(sampled_subg_dict, f)
        print(f"2-hop subgraph dict saved to {output_path}")
    
    # Generate news subgraphs
    output_path = f"news_subgraph-{args.dataset}.pkl"
    if os.path.exists(output_path):
        print(f"News subgraph dict already exists at {output_path}, loading...")
        with open(output_path, 'rb') as f:
            news_subgraphs = pickle.load(f)
    else:
        # Generate news subgraphs
        print("Generating news subgraphs...")
        news_subgraphs = generate_news_subgraphs(linked_entities, args.dataset, induced_subg, sampled_subg_dict, global2local)
        # Save to pickle
        with open(output_path, 'wb') as f:
            pickle.dump(news_subgraphs, f)
        print(f"News subgraph dict saved to {output_path}")
    
    single_component_cnt = sum(1 for g, _ in news_subgraphs.values() if g.num_nodes == 1 and g.edge_index.numel() == 0)
    print(f"Single component news subgraphs: {single_component_cnt}/{len(news_subgraphs)}")

    # Print summary for first 20 entity subgraphs
    print(f"Total news subgraphs: {len(news_subgraphs)}")

    num_nodes_list = [sub.num_nodes if sub is not None else 0 for sub, _ in news_subgraphs.values()]
    print(f"Max num_nodes: {max(num_nodes_list)}")
    print(f"Min num_nodes: {min(num_nodes_list)}")
    print(f"Avg num_nodes: {np.mean(num_nodes_list)}")

    edge_counts = [sub.edge_index.size(1) if sub is not None else 0 for sub, _ in news_subgraphs.values()]
    print(f"Max num_edges: {max(edge_counts)}")
    print(f"Min num_edges: {min(edge_counts)}")
    print(f"Avg num_edges: {np.mean(edge_counts)}")