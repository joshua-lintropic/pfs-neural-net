import torch, numpy as np, gnn, os, glob
from config import *

def to_Graph(class_props: np.ndarray) -> gnn.BipartiteData:
    n_classes   = class_props.shape[0]

    e_h, e_g, edge_attr = [], [], []
    for h in range(nfibers):
        for g in range(n_classes):
            e_h.append(h)
            e_g.append(g)
            edge_attr.append(np.zeros(n_x, dtype=np.float32))

    edge_index = torch.tensor([e_h, e_g], dtype=torch.long)
    edge_attr  = torch.tensor(edge_attr).float()

    x_h = torch.zeros(nfibers, n_h)                    # (nfibers × n_h)
    x_g = torch.tensor(class_props, dtype=torch.float) # (n_g × ?)
    u   = torch.zeros(1, n_u)

    return gnn.BipartiteData(edge_index.cuda(), x_h.cuda(), x_g.cuda(), edge_attr.cuda(), u.cuda())


# This function varies from problem to problem. 
# The following example only works for the PFS problem
# def to_Graph(indices,properties):
#     # properties: galaxy properties useful for g nodes
#     # indices: pre-calculated connectivity
#     properties = np.array(properties)
#     edge_attr = []
#     e_h = [] # start from h nodes
#     e_g = [] # end at g nodes

#     # Graph Connectivity Related to the Problem
#     for i,index in enumerate(indices):
#         for j in range(len(index)):
#             if index[j]<2394: 
#                 e_h.append(index[j])
#                 e_g.append(k)
#                 edge_attr.append(np.zeros(n_x)) # Edge initialization

#     edge_attr = torch.tensor(edge_attr).float()
#     edge_index = torch.tensor([e_h,e_g],dtype=torch.long)

#     x_h = torch.zeros(2394,n_h).float()
#     x_g = torch.tensor(properties[reachable]).float()
#     u=torch.tensor([np.zeros(n_u)]).float()
#     data = gnn.BipartiteData(edge_index.cuda(),x_h.cuda(),x_g.cuda(),edge_attr.cuda(),u.cuda())
#     return data

if __name__ == '__main__':
    # class properties file:  utils‑<case>.txt   (n_g rows)
    # utils = np.loadtxt(f'initial_info/utils-{case}.txt')   # shape (n_g, ≥2)

    utils = np.array([
        [68.2 * 10**3,  2],
        [69.3 * 10**3,  2],
        [96.3 * 10**3,  2],
        [14.4 * 10**3, 12],
        [22.0 * 10**3,  6],
        [ 8.3 * 10**3,  6],
        [14.0 * 10**3, 12],
        [22.0 * 10**3,  6],
        [ 7.4 * 10**3,  3],
        [ 4.5 * 10**3,  6],
        [ 2.8 * 10**3, 12],
        # TODO: each galaxy in Class 12 has indep time req
        [ 9.7 * 10**3, 15]
    ])

    if not os.path.exists('graphs-%s/'%case):
        os.system('mkdir graphs-%s'%case)

    module = {'train': ntrain, 'valid': nvalid, 'test': ntest}
    for t, n in module.items(): 
        for i in range(n):
            graph = to_Graph(utils)
            torch.save(graph, f'graphs-{case}/graph-{i}-{t}.pt')
