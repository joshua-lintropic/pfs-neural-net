import torch
import numpy as np
import gnn

"""
Module for converting fiber-galaxy index and property data into
PyG BipartiteData graphs and saving them to disk.

Each graph represents which galaxies can be observed by which fibers,
as a bipartite graph with separate source (fibers) and target (galaxies) nodes.
"""

# === CONSTANTS ===
NCLASSES = 12
NFIBERS = 2000

def to_Graph(properties):
    """
    Convert index and property arrays into a BipartiteData graph.

    Constructs edge_index, edge_attr, and node features for a bipartite graph
    with fixed number of fibers (source nodes) and variable number of galaxies.

    Args:
        indices (array-like of shape [N_galaxies, N_fibers_per_galaxy]):
            For each galaxy i, a list of fiber indices that can observe it.
        properties (array-like of shape [N_galaxies, F_xs]):
            Feature vectors for each galaxy.
        maxtime (int, optional):
            Maximum observation time (unused placeholder).

    Returns:
        gnn.BipartiteData:
            A PyG Data object containing:
            - x_s: zeros of shape [2394, F_xs] for fiber node features.
            - x_t: tensor of shape [N_reachable, F_xs] for reachable galaxies.
            - edge_index: long tensor [2, E] of source->target edges.
            - edge_attr: float tensor [E, F_e] of zero-initialized edge features.
            - u: global feature tensor of shape [1, F_u].
    """
    properties = np.array(properties)
    edge_attr = []
    e_s = []  # source indices (fibers)
    e_t = []  # target indices (classes)
    k = 0
    reachable = np.zeros(len(properties), dtype=bool)

    # Build complete edges: every fiber->class yields an edge
    for i in range(properties.shape[0]):
        for fiber_idx in range(NFIBERS):
            e_s.append(fiber_idx)
            e_t.append(k)
            edge_attr.append(np.zeros(gnn.F_e))
        reachable[i] = True
        k += 1

    # Convert to tensors and sort edges by source id (fiber)
    edge_attr = np.array(edge_attr)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    edge_index = torch.tensor([e_s, e_t], dtype=torch.long)
    order = torch.argsort(edge_index[0])
    edge_attr = edge_attr[order]
    edge_index = edge_index[:, order]

    # Node features: fibers (zeros) and galaxies (properties of reachable ones)
    x_s = torch.zeros(NCLASSES, gnn.F_xs, dtype=torch.float)
    x_t = torch.tensor(properties[reachable], dtype=torch.float)
    u = torch.zeros(1, gnn.F_u, dtype=torch.float)

    # Create and return the BipartiteData on GPU
    data = gnn.BipartiteData(
        edge_index.cuda(),
        x_s.cuda(),
        x_t.cuda(),
        edge_attr.cuda(),
        u.cuda()
    )
    return data


if __name__ == '__main__':
    """
    Script entrypoint: loads utility properties and fiber-galaxy pair files,
    filters out galaxies with no observing fibers, converts each to a
    BipartiteData graph, and saves to 'graphs/graph-<i>.pt'.
    """
    ngraph = 25
    utils = np.loadtxt('utils.txt')
    
    for igraph in range(ngraph):
        # Build graph and save
        graph = to_Graph(utils)
        torch.save(graph, f'graphs/graph-{igraph}.pt')
