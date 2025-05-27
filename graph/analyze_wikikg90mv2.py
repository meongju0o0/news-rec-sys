import random
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import networkit as nk
from ogb.lsc import WikiKG90Mv2Dataset
from tqdm import tqdm


def build_1hop_subgraph(dataset, input_entity_id):
    """
    Returns:
      nodes: 1D numpy array of node IDs in the 1-hop subgraph
      edges: 2D numpy array of shape (num_edges, 2) of undirected unique edges [u, v]
    """
    # raw triples: [head, rel, tail]
    triples = dataset.train_hrt  # shape (N,3)

    # 1) 1-hop 이웃 수집
    mask_h = (triples[:, 0] == input_entity_id)
    mask_t = (triples[:, 2] == input_entity_id)
    neigh_h = triples[mask_h, 2]
    neigh_t = triples[mask_t, 0]
    neighbors1 = np.unique(np.concatenate([neigh_h, neigh_t]))

    # 2) 노드 리스트: 중심 + 1-hop
    nodes = np.unique(np.concatenate([[input_entity_id], neighbors1]))

    # 3) 서브그래프용 엣지 필터링 (both ends in `nodes`)
    mask_sub = np.isin(triples[:, 0], nodes) & np.isin(triples[:, 2], nodes)
    sub_triples = triples[mask_sub]

    # 4) 엣지 리스트 생성 (undirected, unique)
    pairs = sub_triples[:, [0, 2]]          # (num_sub, 2)
    pairs = np.sort(pairs, axis=1)          # ensure (u,v) with u<=v
    edges = np.unique(pairs, axis=0)        # remove duplicates

    return nodes, edges


def draw_graph(G, input_entity_idx, save_path=None):
    print("Print subgraph generated with given entity's 2-hop neighbors...")

    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, seed=42)
    node_colors = ['pink' if node == input_entity_idx else 'lightskyblue' for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=50)
    nx.draw_networkx_edges(G, pos)
    plt.title(f"2-hop Graph for Entity {input_entity_idx}")
    plt.axis("off")
    
    if save_path is None:
        save_path = f"2_hop_graph_for_entity_{input_entity_idx}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__=='__main__':
    dataset = WikiKG90Mv2Dataset(root = '../dataset')
    
    for _ in tqdm(range(1000)):
        input_entity_idx = dataset.train_hrt[_][0]
        nodes, edges = build_1hop_subgraph(dataset, input_entity_idx)
        print(f"Entity {input_entity_idx} has {len(nodes)} nodes and {len(edges)} edges.")
