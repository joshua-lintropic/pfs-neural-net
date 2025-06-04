"""
Loss for the *counts* formulation   (n_{ik}  instead of   t_{ik})

Output tensor `assign` produced by the GNN lives on every edge and is
interpreted as “how many sources of class g the fibre h will observe”.
During training we keep it continuous in (0, ∞) but push it towards {0,1}
through the usual noisy‑sigmoid trick.

Notation inside the code
------------------------
assign        : n_{ik}′   (edge‑level variable)
timereq[g]    : T_i       (exposure time needed by class i)
totgal[g]     : N_i       (# galaxies in class i)
"""
import torch, torch.nn.functional as F
from torch_scatter import scatter
# from config import Tmax, sharpness, noiselevel

Tmax = 15
sharpness = 20
noiselevel = 0.3 #[0.2,0.3,0.2,0.3][idx]

def Lossfun(assign, graph, penalty=0., finaloutput=False):
    src, tgt  = graph.edge_index
    timereq   = graph.x_g[:, 0] 
    totgal    = graph.x_g[:, 1] 

    if noiselevel > 0:
        noise = torch.rand_like(assign).sub(0.5).mul(noiselevel)
    else:
        noise = 0.
    assign_hat = torch.sigmoid((assign + noise - 0.5) * sharpness)

    class_cnt = scatter(assign_hat, tgt, dim_size=timereq.size(0), reduce='sum')

    time_per_edge = assign_hat * timereq[tgt]
    fibre_time    = scatter(time_per_edge, src,
                             dim_size=graph.x_h.size(0), reduce='sum')

    overtime   = fibre_time - Tmax
    time_term  = penalty * (F.relu(overtime).pow(2)).sum()

    completeness = class_cnt / totgal.clamp_min(1.)   # avoid /0
    util        = completeness.min()

    if finaloutput:
        # TODO: modify sharpness? 
        assign_hat = torch.sigmoid((assign - 0.5) * 1000)
        class_cnt  = scatter(assign_hat, tgt, dim_size=timereq.size(0), reduce='sum')
        completeness = class_cnt / totgal.clamp_min(1.)
        util = completeness.min()

    tot_edges_active = assign_hat.sum()

    return -(util) + time_term, util, tot_edges_active, \
           F.relu(overtime).mean(), F.relu(-overtime).mean()
