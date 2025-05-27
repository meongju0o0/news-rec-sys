import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.utils import to_networkx, to_scipy_sparse_matrix

data = torch.load('entity_2hop_subgraph-small.pt')
edge_index = data.edge_index
num_nodes = data.num_nodes

adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).tocsr()

selected_nodes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
print("Selected nodes:", selected_nodes)

G_nx = to_networkx(data, to_undirected=True)

for node in selected_nodes:
    one_hop = adj[node].nonzero()[1]
    two_hop = adj[one_hop].nonzero()[1]
    two_hop = np.unique(np.concatenate((one_hop, two_hop)))
    
    nodes_within_k = np.unique(np.concatenate(([node], two_hop)))

    subG = G_nx.subgraph(nodes_within_k).copy()

    node_array = np.array(list(subG.nodes()))
    colors = np.full(len(node_array), '#87CEFA', dtype=object)
    center_idx = np.where(node_array == node)[0]
    if center_idx.size > 0:
        colors[center_idx[0]] = '#FFC0CB'

    plt.figure(figsize=(6, 6))
    pos = nx.spring_layout(subG, seed=42)
    nx.draw(subG, pos, with_labels=False, node_color=colors, edge_color='gray', node_size=50)
    plt.title(f"2-hop neighborhood of node {node}")
    plt.savefig(f"subgraph_node_{node}.png", bbox_inches='tight')
    plt.close()

    print(f"Saved subgraph for node {node} as subgraph_node_{node}.png")
