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

def Loss(time, graph, penalty=1.0, finaloutput=False):
    """
    time:     [#edges] predicted t_{ik} for each (fiber k → class i) edge
    graph.x_s:[#fibers, …]
    graph.x_t:[#classes, 2]  columns = [T_i, N_i]
    graph.edge_index: (src=fiber_idx, tgt=class_idx)
    """
    # unpack
    src, tgt = graph.edge_index
    T_i = graph.x_t[:, 0]   # required hours per visit for each class
    N_i = graph.x_t[:, 1]   # total # galaxies in each class
    total_time = 42         # T in your formula

    # --- 1) Compute class‐wise soft visit counts n'_i = (∑_k t_{ik}) / T_i
    class_time = scatter(time, tgt, dim_size=T_i.size(0), reduce='sum')
    n_prime = class_time / (T_i + 1e-6)       # shape [#classes]

    # --- 2) (Soft) rounding → n_soft ≈ round(n_prime)
    #     replace with your noisy‐sigmoid routine; here’s a placeholder:
    sharp = 10.0
    n_soft = torch.sigmoid((n_prime - torch.round(n_prime)) * sharp) * torch.round(n_prime) \
           + (1 - torch.sigmoid((n_prime - torch.round(n_prime)) * sharp)) * torch.floor(n_prime)

    # --- 3) Class‐completeness = n_i / N_i
    completeness = n_soft / (N_i + 1e-6)
    completeness = torch.clamp(completeness, 0.0, 1.0)
    totutils = torch.min(completeness)

    # --- 4) Penalty on per‐fiber overtime
    fiber_time = scatter(time, src, dim_size=graph.x_s.size(0), reduce='sum')
    overtime = fiber_time - total_time
    leaky = nn.LeakyReLU(negative_slope=0.0)  # squared‐leaky‐ReLU: p(x) = (LeakyReLU(x))^2
    penalty_term = penalty * torch.sum(leaky(overtime)**2)

    # --- 5) Final loss
    loss = -totutils + penalty_term

    if finaloutput:
        # optionally produce hard counts & diagnostics
        return loss, totutils, n_prime, fiber_time
    else:
        return loss, totutils

# def Loss(time,graph,penalty=0,finaloutput=False,batchsize=1):
#     print(f"time ({time.shape})", time)
#     total_time = 42 # total number of visits
#     # Hyperparameter
#     leaky = nn.LeakyReLU(0.) # parameter here tunes the penalty difference between overtime and undertime
#     # Batch
#     bt = graph.batch
#     src,tgt = graph.edge_index # should still be fully connected? 
#     print(f"src, tgt (is it fully connected)?", src, tgt)
#     x_s = graph.x_s
#     x_t = graph.x_t
#     timereq = x_s[:,0]
#     tgtclass = x_s[:,1:-1]
#     totclass = torch.sum(tgtclass,dim=0)
#     # Diagnostics
#     print("tgt min, tgt max, x_s", tgt.min().item(), tgt.max().item(), x_s.size(0))
#     print("src min, src max, x_t", src.min().item(), src.max().item(), x_t.size(0))
#     # Time per galaxy
#     time_sum = scatter(time,src,dim_size=x_s.size(0),reduce='sum')
#     # Time per fiber
#     spent_time = scatter(time,tgt,dim_size=x_t.size(0),reduce='sum')
#     # Overtime
#     overtime = spent_time-total_time
#     delta = leaky(overtime).view(1,NCLASSES) 
#     # Penalty
#     time_constraint = torch.sum(penalty*delta*delta)
#     # Reward
#     sharpness = 5
#     observed = torch.sigmoid((time_sum-timereq+0.5)*sharpness)
#     reward = (observed*tgtclass.T).T
#     completeness = torch.sum(reward,dim=0)/torch.sum(tgtclass,dim=0)
#     completeness = torch.nan_to_num(completeness, nan=0.0)
#     totnum = torch.sum(observed)
#     totutils = torch.min(completeness)
    
#     if finaloutput:
#         sharpness = 1000
#         observed = torch.sigmoid((time_sum-timereq+0.5)*sharpness)
#         reward = (observed*tgtclass.T).T
#         clsnum = torch.sum(reward,dim=0)
#         clsall = torch.sum(tgtclass,dim=0)
#         completeness = clsnum/clsall
#         totnum = torch.sum(observed)
#         totutils = torch.min(completeness)
#         #print(clsnum) #number observed in each class
#         #print(clsall) #total number in each class
#     return -totutils+time_constraint,totutils,totnum,torch.sum(F.relu(overtime))/(NCLASSES*total_time),torch.sum(F.relu(-overtime))/(NCLASSES*total_time)

if __name__ == '__main__':
    # SLURM setup
    idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    ID = str(idx)
    print(f"SLURM ID: {ID}")

    # --- Hyperparameters ---
    ntrain, ntest = 25, 0
    batchsize = 1
    sharpness = 20
    noiselevel = [0.2, 0.3, 0.2, 0.3][idx]
    nepoch_pre, nepoch = 1, 10
    lr_pre, lr = 5e-4, [1e-4, 1e-4, 1e-4, 1e-4][idx]
    penalty_pre = 1e-1
    penalty_ini, penalty_end = 1.0, 1.0
    train = True

    batchsize = min(batchsize, ntrain)

    # --- Data loading ---
    utils = np.loadtxt('utils.txt')
    graphs = [torch.load(f'graphs/graph-{i}.pt', weights_only=False) for i in range(ntrain)]
    dataset = g.Loader(graphs_list=graphs)
    dataloader = torch_geometric.loader.DataLoader(dataset, batch_size=batchsize)

    # Precompute batch index tensors
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
        except FileNotFoundError:
            print('No pre-trained checkpoint found.')

        for epoch in range(nepoch_pre):
            for i_batch, graph in enumerate(dataloader):
                gnn.zero_grad()
                time_pred, _ = gnn(graph, train_be[i_batch], train_bs[i_batch], train_bt[i_batch])
                loss, utility = Loss(time_pred, graph, penalty=penalty_pre)
                loss.backward()
                if train:
                    optimizer.step()
                print(f"Pre-Train Batch {i_batch}: Loss={loss.item():.4f}, Utility={utility:.4f}")

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
        except FileNotFoundError:
            print('No checkpoint found, using pre-trained model.')
            gnn.load_state_dict(torch.load('model_gnn_pre' + ID + '.pth'))

        penalty = penalty_ini
        rate = (penalty_end / penalty_ini) ** (1.0 / nepoch)

        for epoch in range(nepoch):
            for i_batch, graph in enumerate(dataloader):
                gnn.zero_grad()
                time_pred, _ = gnn(graph, train_be[i_batch], train_bs[i_batch], train_bt[i_batch])
                loss, utility = Loss(time_pred, graph, penalty=penalty)
                loss.backward()
                if train:
                    optimizer.step()
                print(f"Train Batch {i_batch}: Loss={loss.item():.4f}, Utility={utility:.4f}")
            penalty *= rate

        torch.save(gnn.state_dict(), 'model_gnn' + ID + '.pth')
        print('Training Finished')

        # --- Evaluation ---
        print('Evaluation Results:')
        for i_batch, graph in enumerate(dataloader):
            time_pred, _ = gnn(graph, train_be[i_batch], train_bs[i_batch], train_bt[i_batch], train=False)
            loss, utility = Loss(time_pred, graph, penalty=penalty)
            print(f"Eval Batch {i_batch}: Loss={loss.item():.4f}, Utility={utility:.4f}")

# if __name__ == '__main__':
#     """
#     Main training script entrypoint.

#     Loads graphs from disk, constructs batched DataLoader, and runs:
#       1) Optional pre-training phase over nepoch_pre epochs.
#       2) Main training phase over nepoch epochs,
#          gradually increasing overtime penalty.
#     Saves GNN model checkpoints and prints per-batch metrics.

#     Expects environment variable SLURM_ARRAY_TASK_ID to index hyperparameters.
#     """
#     idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
#     ID = str(idx)
#     print(f"SLURM ID: {ID}")

#     # --- Hyperparameter setup ---
#     ntrain = 25
#     ntest = 0
#     batchsize = 1
#     sharpness = 20
#     noiselevel = [0.2, 0.3, 0.2, 0.3][idx]
#     train = True
#     nepoch_pre = 1
#     nepoch = 10
#     lr_pre = 5e-4
#     penalty_pre = 1e-1
#     lr = [1e-4, 1e-4, 1e-4, 1e-4][idx]
#     penalty_ini = 1.0
#     penalty_end = 1.0

#     # Prevent batchsize > ntrain
#     batchsize = min(batchsize, ntrain)

#     # --- Data loading ---
#     utils = np.loadtxt('utils.txt')
#     graphs = [torch.load(f'graphs/graph-{i}.pt', weights_only=False) for i in range(ntrain)]
#     dataset = g.Loader(graphs_list=graphs)
#     dataloader = torch_geometric.loader.DataLoader(dataset, batch_size=batchsize)

#     # Precompute batch index tensors for edges/nodes
#     train_be, train_bs, train_bt = [], [], []
#     for graph in dataloader:
#         E, Ns, Nt = graph.edge_attr.size(0), graph.x_s.size(0), graph.x_t.size(0)
#         train_be.append(torch.zeros(E, dtype=torch.long).cuda())
#         train_bs.append(torch.zeros(Ns, dtype=torch.long).cuda())
#         train_bt.append(torch.zeros(Nt, dtype=torch.long).cuda())

#     # --- Pre-training loop ---
#     if nepoch_pre > 0:
#         print('Start Pre-Training')
#         gnn = g.GNN().cuda()
#         gnn.sharpness = sharpness
#         gnn.noiselevel = noiselevel
#         optimizer = torch.optim.Adam(gnn.parameters(), lr=lr_pre)
#         try:
#             gnn.load_state_dict(torch.load('model_gnn_pre' + ID + '.pth'))
#         except:
#             print('No pre-trained checkpoint found.')
#         for epoch in range(nepoch_pre):
#             for i_batch, graph in enumerate(dataloader):
#                 gnn.zero_grad()
#                 print("\nTRAIN.PY")
#                 print(f"i_batch:", i_batch)
#                 print(f"train_be ({len(train_be[i_batch])}):", train_be[i_batch])
#                 print(f"train_bs ({len(train_bs[i_batch])}):", train_bs[i_batch])
#                 print(f"train_bt ({len(train_bt[i_batch])}):", train_bt[i_batch])
#                 print()
#                 time_pred, _ = gnn(graph, train_be[i_batch], train_bs[i_batch], train_bt[i_batch])
#                 loss, gu, nu, ot, ut = Loss(time_pred, graph, penalty=penalty_pre, batchsize=batchsize)
#                 loss.backward()
#                 print(f"Batch {i_batch}: -U={-loss.item():.1f}, G={gu:.1f}, N={nu:.1f}, +OT={ot:.1f}, +UT={ut:.1f}")
#                 if train:
#                     optimizer.step()
#         torch.save(gnn.state_dict(), 'model_gnn_pre' + ID + '.pth')
#         print('Pre-Training Finished')

#     # --- Main training loop ---
#     if nepoch > 0:
#         print('Start Training')
#         gnn = g.GNN().cuda()
#         gnn.sharpness = sharpness
#         gnn.noiselevel = noiselevel
#         optimizer = torch.optim.Adam(gnn.parameters(), lr=lr)
#         try:
#             gnn.load_state_dict(torch.load('model_gnn' + ID + '.pth'))
#         except:
#             print('No checkpoint found, using pre-trained model.')
#             gnn.load_state_dict(torch.load('model_gnn_pre' + ID + '.pth'))
#         penalty = penalty_ini
#         rate = (penalty_end / penalty_ini) ** (1.0 / nepoch)
#         for epoch in range(nepoch):
#             for i_batch, graph in enumerate(dataloader):
#                 gnn.zero_grad()
#                 time_pred, _ = gnn(graph, train_be[i_batch], train_bs[i_batch], train_bt[i_batch])
#                 loss, gu, nu, ot, ut = Loss(time_pred, graph, penalty=penalty, batchsize=batchsize)
#                 loss.backward()
#                 if train:
#                     optimizer.step()
#                 print(f"Batch {i_batch}: -U={-loss.item():.1f}, G={gu:.1f}, N={nu:.1f}, +OT={ot:.1f}, +UT={ut:.1f}")
#             penalty *= rate
#         torch.save(gnn.state_dict(), 'model_gnn' + ID + '.pth')
#         print('Training Finished')

#         # --- Evaluation ---
#         print('Evaluation Results:')
#         for i_batch, graph in enumerate(dataloader):
#             time_pred, _ = gnn(graph, train_be[i_batch], train_bs[i_batch], train_bt[i_batch], train=False)
#             loss, gu, nu, ot, ut = Loss(time_pred, graph, penalty=penalty, batchsize=batchsize)
#             print(f"Batch {i_batch}: -U={-loss.item():.1f}, G={gu:.1f}, N={nu:.1f}, +OT={ot:.1f}, +UT={ut:.1f}")
