import numpy as np
import torch
import torch_scatter
import torch_geometric
import gnn as g
import os

# total number of exposures per fiber
TOTAL_TIME = 6           # per‑fiber time budget  ( “T” in the paper )
NCLASSES = 12            # temperature for the differentiable min
SOFTMIN_T = 5.0          # λ for the time‑constraint term
DEFAULT_LMB = 1.0        # numerical safety
EPS = 1E-6

import torch
import torch.nn as nn
from torch_scatter import scatter_mean,scatter
import torch.nn.functional as F

def Loss(time,graph,penalty=0,finaloutput=False,batchsize=1):
    total_time = 42 # total number of visits
    # Hyperparameter
    leaky = nn.LeakyReLU(0.) # parameter here tunes the panelty difference between overtime and undertime
    # Batch
    bt = graph.batch
    src,tgt = graph.edge_index
    x_s = graph.x_s
    x_t = graph.x_t
    timereq = x_s[:,0]
    tgtclass = x_s[:,1:-1]
    totclass = torch.sum(tgtclass,dim=0)
    # Diagnostics
    print("tgt min, tgt max, x_s", tgt.min().item(), tgt.max().item(), x_s.size(0))
    print("src min, src max, x_t", src.min().item(), src.max().item(), x_t.size(0))
    # Time per galaxy
    time_sum = scatter(time,src,dim_size=x_s.size(0),reduce='sum')
    # Time per fiber
    spent_time = scatter(time,tgt,dim_size=x_t.size(0),reduce='sum')
    # Overtime
    overtime = spent_time-total_time
    delta = leaky(overtime).view(1,NCLASSES) 
    # Penalty
    time_constraint = torch.sum(penalty*delta*delta)
    # Reward
    sharpness = 5
    observed = torch.sigmoid((time_sum-timereq+0.5)*sharpness)
    reward = (observed*tgtclass.T).T
    completeness = torch.sum(reward,dim=0)/torch.sum(tgtclass,dim=0)
    totnum = torch.sum(observed)
    totutils = torch.min(completeness)
    
    if finaloutput:
        sharpness = 1000
        observed = torch.sigmoid((time_sum-timereq+0.5)*sharpness)
        reward = (observed*tgtclass.T).T
        clsnum = torch.sum(reward,dim=0)
        clsall = torch.sum(tgtclass,dim=0)
        completeness = clsnum/clsall
        totnum = torch.sum(observed)
        totutils = torch.min(completeness)
        #print(clsnum) #number observed in each class
        #print(clsall) #total number in each class
    return -totutils+time_constraint,totutils,totnum,torch.sum(F.relu(overtime))/(NCLASSES*total_time),torch.sum(F.relu(-overtime))/(NCLASSES*total_time)

# def Loss(time, graph, penalty=0, batchsize=1):
#     """
#     Compute the training loss for a single bipartite graph schedule.

#     The loss is defined as -total_utility + time_constraint_penalty, where:
#       - total_utility = sum of per-galaxy utilities + global coverage utility
#       - time_constraint_penalty = penalty * sum of squared overtime per fiber

#     Args:
#         time (torch.Tensor[E]):   Scheduled observation time per edge.
#         graph (gnn.BipartiteData): Graph containing attributes:
#             edge_index, x_s (fiber features), x_t (class features), batch index.
#         penalty (float):    Coefficient for penalizing overtime squared.
#         batchsize (int):    Number of graphs in batch (currently only 1 supported).

#     Returns:
#         Tuple of five torch.Tensor:
#           - loss (scalar): negative utility plus penalty.
#           - tot_gal_utils (scalar): sum of piecewise-linear per-galaxy utilities.
#           - tot_num_utils (scalar): coverage-based global utility.
#           - total_overtime (scalar): sum of positive overtime across fibers.
#           - total_undertime (scalar): sum of positive undertime across fibers.
#     """
#     # Activation for asymmetric overtime penalty
#     leaky = torch.nn.LeakyReLU(1.0)

#     src, tgt = graph.edge_index
#     x_t = graph.x_t       # [N_class, F_glob]
#     x_s = graph.x_s       # [N_fiber, F_xs]
#     bt = graph.batch      # [E] mapping edges -> graph id

#     # 1) Aggregate scheduled time per galaxy and per fiber
#     time_sum = torch_scatter.scatter(time, tgt, dim=0, dim_size=x_t.size(0), reduce='sum')
#     spent_time = torch_scatter.scatter(time, src, dim=0, dim_size=x_s.size(0), reduce='sum')

#     # 2) Overtime penalty (ReLU-based) per fiber
#     overtime = spent_time - TOTAL_TIME
#     delta = leaky(overtime).view(batchsize, -1)
#     time_constraint = penalty * torch.sum(delta * delta)

#     # 3) Per-galaxy piecewise-linear utilities
#     reward = x_t  # assume columns encode piecewise coefficients and coverage flag
#     galutils = (
#         time_sum * reward[:, 0]
#         + torch.nn.functional.relu(time_sum - 1.0) * (reward[:, 1] - 2 * reward[:, 0])
#         + torch.nn.functional.relu(time_sum - 2.0) * (reward[:, 2] - 2 * reward[:, 1] + reward[:, 0])
#         + torch.nn.functional.relu(time_sum - 3.0) * (reward[:, 3] - 2 * reward[:, 2] + reward[:, 1])
#         - torch.nn.functional.relu(time_sum - 4.0) * reward[:, 3]
#     )
#     tot_gal_utils = torch.sum(galutils)

#     # 4) Global coverage utility: encourage >=1 visit for desired galaxies
#     requirement = reward[:, 4]
#     coverage = torch.sigmoid(5 * (time_sum - 0.5)) * requirement
#     num_covered = torch_scatter.scatter(coverage, bt, dim=0, reduce='sum')
#     numreq = 5000
#     avgrwd = 2.0
#     num_utils = avgrwd * numreq * torch.sigmoid(0.01 * (num_covered - numreq))
#     tot_num_utils = torch.sum(num_utils)

#     # 5) Total utility and final loss
#     tot_utils = tot_gal_utils + tot_num_utils
#     total_overtime = torch.sum(torch.nn.functional.relu(overtime))
#     total_undertime = torch.sum(torch.nn.functional.relu(-overtime))
#     loss = -tot_utils + time_constraint

#     return loss, tot_gal_utils, tot_num_utils, total_overtime, total_undertime

# def Loss(time, graph, penalty: float = DEFAULT_LMB, batchsize: int = 1):
#     """
#     Differentiable surrogate of
#         L = - min_i  n_i / N_i  +  λ Σ_k p( Σ_i n_ik T_i − T )

#     Parameters
#     ----------
#     time : (E,) tensor
#         Continuous edge “time” variables n_ik * T_i produced by the network
#         (the noisy‑sigmoid relaxation of the rounded integer in the paper).
#     graph : gnn.BipartiteData
#         Must contain for every class‑node j
#             graph.x_t[j, 0] = T_i   (exposure time per galaxy of that class)
#             graph.x_t[j, 1] = N_i   (survey target for that class)
#         as well as standard fields edge_index, x_s, batch.
#     penalty : float, default = DEFAULT_LMB
#         Multiplier λ for the overtime penalty.
#     batchsize : int, default = 1
#         Number of independent graphs in the batch.

#     Returns
#     -------
#     loss                -- scalar      : the value optimised by SGD
#     min_completeness    -- scalar      :   min_i n_i / N_i  (after soft‑min)
#     total_overtime      -- scalar
#     total_undertime     -- scalar
#     """
#     leaky = torch.nn.LeakyReLU(1.0)        # asymmetric overtime slope

#     src, tgt = graph.edge_index
#     T_i = graph.x_t[:, 0].clamp_min(EPS)   # (N_cls,)
#     N_i = graph.x_t[:, 1].clamp_min(EPS)   # (N_cls,)

#     # ------------------------------------------------------------------
#     # 1)   Aggregate edge “time” to   Σ_k n_ik T_i    and per‑fiber load
#     # ------------------------------------------------------------------
#     time_sum_cls = torch_scatter.scatter(time, tgt, dim=0, dim_size=T_i.size(0), reduce='sum')
#     time_sum_fib = torch_scatter.scatter(time, src, dim=0, dim_size=graph.x_s.size(0), reduce='sum')

#     # ------------------------------------------------------------------
#     # 2)   Overtime / undertime penalty        (same as original code)
#     # ------------------------------------------------------------------
#     overtime = time_sum_fib - TOTAL_TIME                      # (N_fibers,)
#     delta    = leaky(overtime).view(batchsize, -1)
#     time_constraint = penalty * torch.sum(delta * delta)

#     # ------------------------------------------------------------------
#     # 3)   Class‑level counts   n_i  and completeness  c_i
#     # ------------------------------------------------------------------
#     n_i = time_sum_cls / T_i                                  # galaxies scheduled
#     completeness = n_i / N_i                                  # want to maximise min

#     # differentiable soft‑minimum   min_i c_i  ≈  softmin(c_i)
#     soft_min = -torch.logsumexp(-SOFTMIN_T * completeness, dim=0) / SOFTMIN_T

#     # ------------------------------------------------------------------
#     # 4)   Final loss (negate because we maximise completeness)
#     # ------------------------------------------------------------------
#     loss = -soft_min + time_constraint

#     total_overtime  = torch.sum(torch.nn.functional.relu(overtime))
#     total_undertime = torch.sum(torch.nn.functional.relu(-overtime))

#     return loss, soft_min, total_overtime, total_undertime


if __name__ == '__main__':
    """
    Main training script entrypoint.

    Loads graphs from disk, constructs batched DataLoader, and runs:
      1) Optional pre-training phase over nepoch_pre epochs.
      2) Main training phase over nepoch epochs,
         gradually increasing overtime penalty.
    Saves GNN model checkpoints and prints per-batch metrics.

    Expects environment variable SLURM_ARRAY_TASK_ID to index hyperparameters.
    """
    idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    ID = str(idx)
    print(f"SLURM ID: {ID}")

    # --- Hyperparameter setup ---
    ntrain = 25
    ntest = 0
    batchsize = 1
    sharpness = 20
    noiselevel = [0.2, 0.3, 0.2, 0.3][idx]
    train = True
    nepoch_pre = 1
    nepoch = 10
    lr_pre = 5e-4
    penalty_pre = 1e-1
    lr = [1e-4, 1e-4, 1e-4, 1e-4][idx]
    penalty_ini = 1.0
    penalty_end = 1.0

    # Prevent batchsize > ntrain
    batchsize = min(batchsize, ntrain)

    # --- Data loading ---
    utils = np.loadtxt('utils.txt')
    graphs = [torch.load(f'graphs/graph-{i}.pt', weights_only=False) for i in range(ntrain)]
    dataset = g.Loader(graphs_list=graphs)
    dataloader = torch_geometric.loader.DataLoader(dataset, batch_size=batchsize)

    # Precompute batch index tensors for edges/nodes
    train_be, train_bs, train_bt = [], [], []
    for graph in dataloader:
        E, Ns, Nt = graph.edge_attr.size(0), graph.x_s.size(0), graph.x_t.size(0)
        train_be.append(torch.zeros(E, dtype=torch.long).cuda())
        train_bs.append(torch.zeros(Ns, dtype=torch.long).cuda())
        train_bt.append(torch.zeros(Nt, dtype=torch.long).cuda())

    # --- Pre-training loop ---
    if nepoch_pre > 0:
        print('Start Pre-Training')
        gnn = g.GNN().cuda()
        gnn.sharpness = sharpness
        gnn.noiselevel = noiselevel
        optimizer = torch.optim.Adam(gnn.parameters(), lr=lr_pre)
        try:
            gnn.load_state_dict(torch.load('model_gnn_pre' + ID + '.pth'))
        except:
            print('No pre-trained checkpoint found.')
        for epoch in range(nepoch_pre):
            for i_batch, graph in enumerate(dataloader):
                gnn.zero_grad()
                print("\nTRAIN.PY")
                print(f"i_batch:", i_batch)
                print(f"train_be ({len(train_be[i_batch])}):", train_be[i_batch])
                print(f"train_bs ({len(train_bs[i_batch])}):", train_bs[i_batch])
                print(f"train_bt ({len(train_bt[i_batch])}):", train_bt[i_batch])
                print()
                time_pred, _ = gnn(graph, train_be[i_batch], train_bs[i_batch], train_bt[i_batch])
                loss, gu, nu, ot, ut = Loss(time_pred, graph, penalty=penalty_pre, batchsize=batchsize)
                loss.backward()
                print(f"Batch {i_batch}: -U={-loss.item():.1f}, G={gu:.1f}, N={nu:.1f}, +OT={ot:.1f}, +UT={ut:.1f}")
                if train:
                    optimizer.step()
        torch.save(gnn.state_dict(), 'model_gnn_pre' + ID + '.pth')
        print('Pre-Training Finished')

    # --- Main training loop ---
    if nepoch > 0:
        print('Start Training')
        gnn = g.GNN().cuda()
        gnn.sharpness = sharpness
        gnn.noiselevel = noiselevel
        optimizer = torch.optim.Adam(gnn.parameters(), lr=lr)
        try:
            gnn.load_state_dict(torch.load('model_gnn' + ID + '.pth'))
        except:
            print('No checkpoint found, using pre-trained model.')
            gnn.load_state_dict(torch.load('model_gnn_pre' + ID + '.pth'))
        penalty = penalty_ini
        rate = (penalty_end / penalty_ini) ** (1.0 / nepoch)
        for epoch in range(nepoch):
            for i_batch, graph in enumerate(dataloader):
                gnn.zero_grad()
                time_pred, _ = gnn(graph, train_be[i_batch], train_bs[i_batch], train_bt[i_batch])
                loss, gu, nu, ot, ut = Loss(time_pred, graph, penalty=penalty, batchsize=batchsize)
                loss.backward()
                if train:
                    optimizer.step()
                print(f"Batch {i_batch}: -U={-loss.item():.1f}, G={gu:.1f}, N={nu:.1f}, +OT={ot:.1f}, +UT={ut:.1f}")
            penalty *= rate
        torch.save(gnn.state_dict(), 'model_gnn' + ID + '.pth')
        print('Training Finished')

        # --- Evaluation ---
        print('Evaluation Results:')
        for i_batch, graph in enumerate(dataloader):
            time_pred, _ = gnn(graph, train_be[i_batch], train_bs[i_batch], train_bt[i_batch], train=False)
            loss, gu, nu, ot, ut = Loss(time_pred, graph, penalty=penalty, batchsize=batchsize)
            print(f"Batch {i_batch}: -U={-loss.item():.1f}, G={gu:.1f}, N={nu:.1f}, +OT={ot:.1f}, +UT={ut:.1f}")
