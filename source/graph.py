import torch
import numpy as np
import gnn
from params import *

"""
Module for converting fiber-galaxy index and property data into
PyG BipartiteData graphs and saving them to disk.

Each graph represents which galaxies can be observed by which fibers,
as a bipartite graph with separate source (fibers) and target (galaxies) nodes.
"""

def to_Graph(properties):
    """
    Convert index and property arrays into a BipartiteData graph.

    Constructs edge_index, edge_attr, and node features for a bipartite graph
    with fixed number of fibers (source nodes) and variable number of galaxies.

    Args:
        properties (array-like of shape [NCLASSES, F]):
            Feature vectors for each galaxy.

    Returns:
        gnn.BipartiteData:
            A PyG Data object containing:
            - x_s: zeros of shape [NFIBERS, F_xs] for fiber node features.
            - x_t: tensor of shape [NCLASSES, F_xs] for reachable classes.
            - edge_index: long tensor [2, E] of source->target edges.
            - edge_attr: float tensor [E, F_e] of zero-initialized edge features.
            - u: global feature tensor of shape [1, F_u].
    """
    properties = np.array(properties)
    edge_attr = []
    e_s = []  # source indices (fibers)
    e_t = []  # target indices (classes)
    # reachable = np.ones(properties.shape[0], dtype=bool) # always reachable

    # Build complete edges: every fiber->class yields an edge
    for i in range(properties.shape[0]):
        for fiber_idx in range(NFIBERS):
            e_s.append(fiber_idx)
            e_t.append(i)
            edge_attr.append(np.zeros(gnn.Fdim))
        # reachable[i] = True
    # print(f"e_t ({len(e_t)}):", set(e_t))

    # Convert to tensors and sort edges by source id (fiber)
    edge_attr = np.array(edge_attr)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    edge_index = torch.tensor([e_s, e_t], dtype=torch.long)
    order = torch.argsort(edge_index[0])
    edge_attr = edge_attr[order]
    # print(f"edge_attr ({edge_attr.shape}):", edge_attr)
    edge_index = edge_index[:, order]

    # Node features: fibers (zeros) and galaxies (properties of reachable ones)
    x_s = torch.randn(NFIBERS, gnn.Fdim, dtype=torch.float)
    x_t = torch.tensor(properties, dtype=torch.float)
    u = torch.zeros(1, gnn.Fdim, dtype=torch.float)

    # Create and return the BipartiteData on GPU
    data = gnn.BipartiteData(
        edge_index,
        x_s,
        x_t,
        edge_attr,
        u
    )
    return data


if __name__ == '__main__':
    """
    Script entrypoint: loads utility properties and fiber-galaxy pair files,
    filters out galaxies with no observing fibers, converts each to a
    BipartiteData graph, and saves to '../graphs/graph-<i>.pt'.
    """
    ngraph = 1
    utils = np.loadtxt('../data/utils.txt')
    utils = np.hstack((utils, np.zeros((utils.shape[0], Fdim - utils.shape[1]))))
    
    for igraph in range(ngraph):
        # Build graph and save
        graph = to_Graph(utils)
        torch.save(graph, f'../graphs/graph-{igraph}.pt')
