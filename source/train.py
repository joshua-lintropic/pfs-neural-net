import numpy as np
import torch
import torch.nn as nn
import torch_geometric
from torch_scatter import scatter
import gnn as g
import os
from params import *
from format import bcolors

def Loss(time, graph, properties, sharpness=10.0, penalty=1.0, finaloutput=False):
    """
    time: [NCLASSES x NFIBERS], predicted t_{ik} for each fiber k, class i 
    properties: [NCLASSES, F_xt] where col1 is T_i and col2 is N_i
    graph.edge_index: (src=fiber_idx, tgt=class_idx)
    """
    # unpack
    src, tgt = graph.edge_index
    T_i = properties[:, 0]   # required hours per visit for each class
    N_i = properties[:, 1]   # total # galaxies in each class

    # compute class‐wise soft visit counts n_i'
    class_time = scatter(time, tgt, dim_size=T_i.size(0), reduce='sum')
    n_prime = class_time / (T_i + 1e-6) # shape [NCLASSES]

    # soft rounding
    n_soft = torch.floor(n_prime) + torch.sigmoid(sharpness * (n_prime - torch.floor(n_prime) - 0.5))

    # class‐completeness = n_i / N_i
    completeness = n_soft / (N_i + 1e-6)
    completeness = torch.clamp(completeness, 0.0, 1.0)
    totutils = torch.min(completeness)

    # penalty on per‐fiber overtime
    fiber_time = scatter(time, src, dim_size=graph.x_s.size(0), reduce='sum')
    overtime = fiber_time - TOTAL_TIME
    leaky = nn.ReLU()  # squared‐leaky‐ReLU: p(x) = (LeakyReLU(x))^2
    penalty_term = penalty * torch.sum(leaky(overtime)**2)

    # final loss
    loss = -totutils + penalty_term

    if finaloutput:
        # optionally produce hard counts & diagnostics
        return loss, totutils, n_prime, fiber_time
    else:
        return loss, totutils

if __name__ == '__main__':
    # SLURM setup
    idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    ID = str(idx)

    # data loading 
    utils = np.loadtxt('../data/utils.txt')
    properties = torch.tensor(utils, dtype=torch.float).to(device)
    graphs = [torch.load(f'../graphs/graph-{i}.pt', weights_only=False) for i in range(ntrain)]
    dataset = g.Loader(graphs_list=graphs)
    dataloader = torch_geometric.loader.DataLoader(dataset, batch_size=batchsize)

    # precompute batch index tensors
    train_be, train_bs, train_bt = [], [], []
    for graph in dataloader:
        E, Ns, Nt = graph.edge_attr.size(0), graph.x_s.size(0), graph.x_t.size(0)
        train_be.append(torch.zeros(E, dtype=torch.long).to(device))
        train_bs.append(torch.zeros(Ns, dtype=torch.long).to(device))
        train_bt.append(torch.zeros(Nt, dtype=torch.long).to(device))

    # pre-training loop ---
    if nepoch_pre > 0:
        print(f'{bcolors.HEADER}STATUS: Start Pre-Training{bcolors.ENDC}')
        gnn = g.GNN().to(device)
        gnn.sharpness = sharpness
        gnn.noiselevel = noiselevel
        optimizer = torch.optim.Adam(gnn.parameters(), lr=lr_pre)
        try:
            gnn.load_state_dict(torch.load('../models/model_gnn_pre' + ID + '.pth'))
        except FileNotFoundError:
            print(f'{bcolors.WARNING}STATUS: No pre-trained checkpoint found.{bcolors.ENDC}')

        for epoch in range(nepoch_pre):
            for i_batch, graph in enumerate(dataloader):
                gnn.zero_grad()
                time_pred, _ = gnn(graph, train_be[i_batch], train_bs[i_batch], train_bt[i_batch])
                loss, utility = Loss(time_pred, graph, properties, sharpness=gnn.sharpness, penalty=penalty_pre)
                loss.backward()
                if train:
                    optimizer.step()
                print(f"OUTPUT: Pre-Train Batch {i_batch}: Loss={loss.item():.4e}, Utility={utility:.4e}")

        torch.save(gnn.state_dict(), '../models/model_gnn_pre' + ID + '.pth')
        print(f'{bcolors.HEADER}STATUS: Pre-Training Finished{bcolors.ENDC}')

    # main training loop 
    if nepoch > 0:
        print(f'{bcolors.OKBLUE}STATUS: Start Training{bcolors.ENDC}')
        gnn = g.GNN().to(device)
        gnn.sharpness = sharpness
        gnn.noiselevel = noiselevel
        optimizer = torch.optim.Adam(gnn.parameters(), lr=lr)
        try:
            gnn.load_state_dict(torch.load('../models/model_gnn' + ID + '.pth'))
        except FileNotFoundError:
            print(f'{bcolors.WARNING}STATUS: No checkpoint found, using pre-trained model.{bcolors.ENDC}')
            gnn.load_state_dict(torch.load('../models/model_gnn_pre' + ID + '.pth'))

        penalty = penalty_ini
        rate = (penalty_end / penalty_ini) ** (1.0 / nepoch)

        for epoch in range(nepoch):
            for i_batch, graph in enumerate(dataloader):
                gnn.zero_grad()
                time_pred, _ = gnn(graph, train_be[i_batch], train_bs[i_batch], train_bt[i_batch])
                loss, utility = Loss(time_pred, graph, properties, sharpness=gnn.sharpness, penalty=penalty)
                loss.backward()
                if train:
                    optimizer.step()
                print(f"OUTPUT: Train Batch {i_batch}: Loss={loss.item():.4e}, Utility={utility:.4e}")
            penalty *= rate

        torch.save(gnn.state_dict(), '../models/model_gnn' + ID + '.pth')
        print(f'{bcolors.OKBLUE}STATUS: Training Finished{bcolors.ENDC}')

        # evaluation 
        print(f'{bcolors.OKGREEN}STATUS: Evaluation Results:{bcolors.ENDC}')
        for i_batch, graph in enumerate(dataloader):
            time_pred, _ = gnn(graph, train_be[i_batch], train_bs[i_batch], train_bt[i_batch], train=False)
            loss, utility = Loss(time_pred, graph, properties, penalty=penalty)
            print(f"{bcolors.BOLD}OUTPUT: Eval Batch {i_batch}: Loss={loss.item():.4e}, Utility={utility:.4e}{bcolors.ENDC}")
