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
    timereq = graph.x_g[:, 0] 
    totgalx = graph.x_g[:, 1] 

    if noiselevel > 0:
        noise = torch.rand_like(assign).sub(0.5).mul(noiselevel)
    else:
        noise = 0.
    assign_hat = torch.sigmoid((assign + noise - 0.5) * sharpness)

    # calculate amount of flow to each class
    class_cnt = scatter(assign_hat, tgt, dim_size=timereq.size(0), reduce='sum')
    time_per_edge = assign_hat * timereq[tgt]
    fibre_time = scatter(time_per_edge, src, dim_size=graph.x_h.size(0), reduce='sum')

    # square ReLU penalty
    overtime   = fibre_time - Tmax
    time_term  = penalty * (F.relu(overtime).pow(2)).sum()

    # determine objective
    completeness = class_cnt / totgalx.clamp_min(1.) # avoid /0
    util = completeness.min()

    if finaloutput:
        # TODO: modify sharpness? 
        assign_hat = torch.sigmoid((assign - 0.5) * 1000)
        class_cnt  = scatter(assign_hat, tgt, dim_size=timereq.size(0), reduce='sum')
        completeness = class_cnt / totgalx.clamp_min(1.)
        util = completeness.min()

    tot_edges_active = assign_hat.sum()

    return -(util) + time_term, util, tot_edges_active, F.relu(overtime).mean(), F.relu(-overtime).mean()
