import pickle
import os
import csv
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch_geometric.utils import to_networkx
from tqdm import tqdm

input_pkl_path = 'entity_2hop_subgraph-small.pkl'
output_img_dir = 'sampled_subgraph_images'

# 1. Load subgraph dictionary
with open(input_pkl_path, 'rb') as f:
    pcst_dict = pickle.load(f)
print(f"Loaded PCST subgraphs for {len(pcst_dict)} entities.")

# 2. Load ent2idx mapping
df = pd.read_csv(
    '../dataset/wikikg90mv2_mapping/entity.csv',
    usecols=['idx', 'entity'],
    dtype={'idx': int, 'entity': str},
    encoding='utf-8',
    engine='c',
    quoting=csv.QUOTE_MINIMAL,
)
ent2idx = dict(zip(df['entity'], df['idx']))
print(f"Loaded entity to index mapping for {len(ent2idx)} entities.")

# 3. Visualize first 10 subgraphs
for wid, data in tqdm(pcst_dict.items()):
    if data is None:
        continue

    # 중심 노드 인덱스 찾기
    global_ids = data.global_node_idx
    target_gid = ent2idx.get(wid, -1)
    idx_arr = np.where(global_ids == target_gid)[0]

    if len(idx_arr) == 0:
        continue
    center_idx = idx_arr[0]

    # 그래프 변환 및 시각화
    G_nx = to_networkx(data, to_undirected=True)
    nodes = np.array(list(G_nx.nodes()))
    colors = np.full(len(nodes), '#87CEFA', dtype=object)
    colors[center_idx] = '#FFC0CB'

    plt.figure(figsize=(6, 6))
    pos = nx.spring_layout(G_nx, seed=42)
    nx.draw(G_nx, pos, with_labels=False, node_color=colors, edge_color='gray', node_size=500)
    plt.title(f"PCST subgraph for {wid}")
    save_path = os.path.join(output_img_dir, f"pcst_subgraph_{wid}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
