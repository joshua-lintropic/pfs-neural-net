import torch
import torch.nn as nn
from torch_scatter import scatter_mean,scatter
import torch.nn.functional as F
# from config import Tmax, sharpness, noiselevel

Tmax = 15
sharpness = 20
noiselevel = 0.3 #[0.2,0.3,0.2,0.3][idx]

def Lossfun(assign, graph, penalty=0., finaloutput=False):
    bt = graph.batch
    src, tgt = graph.edge_index
    x_g = graph.x_g # class features

    time_req = x_g[:, -1] # T_i for each class
    galaxies = x_g[:, 0] # N_i for each class

    noise = torch.rand_like(assign).sub(0.5).mul(noiselevel) if noiselevel > 0 else 0.
    assign_hat = torch.sigmoid((assign + noise - 0.5) * sharpness)

    # calculate amount of flow to each class
    class_count = scatter(assign_hat, tgt, dim_size=time_req.size(0), reduce='sum')
    time_per_edge = assign_hat * time_req[tgt]
    fibre_time = scatter(time_per_edge, src, dim_size=graph.x_h.size(0), reduce='sum')

    # square ReLU penalty
    overtime = fibre_time - Tmax
    time_term = penalty * (F.relu(overtime).pow(2)).sum()

    # determine objective
    completeness = class_count / galaxies.clamp_min(1.) # avoid /0
    util = completeness.min()

    if finaloutput:
        # TODO: modify sharpness? 
        assign_hat = torch.sigmoid((assign - 0.5) * 1000)
        class_count = scatter(assign_hat, tgt, dim_size=time_req.size(0), reduce='sum')
        completeness = class_count / galaxies.clamp_min(1.)
        util = completeness.min()

    tot_edges_active = assign_hat.sum()

    return -(util) + time_term, util, tot_edges_active, F.relu(overtime).mean(), F.relu(-overtime).mean()
