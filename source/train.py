import numpy as np
import torch
import torch.nn as nn
from torch_scatter import scatter
from gnn import GNN, BipartiteData
from params import *

import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import trange
import os

# === DEVICE SPEFICIATIONS ===
ncores = os.cpu_count() or 1
os.environ['OMP_NUM_THREADS'] = str(ncores)
os.environ['MKL_NUM_THREADS'] = str(ncores)
torch.set_num_threads(ncores)
torch.set_num_interop_threads(ncores)

def softround(x, sharpness=20, noiselevel=0.3):
    noise = noiselevel * (torch.rand_like(x) - 0.5)
    x = x + noise
    intpart = torch.floor(x)
    x = intpart + torch.sigmoid(sharpness * (x - 0.5 - intpart))
    return x

def loss_function(graph, class_info, pclass=0.1, pfiber=1.0, finaloutput=False):
    """
    time: [NCLASSES x NFIBERS], predicted t_{ik} for each fiber k, class i 
    properties: [NCLASSES, F_xt] where col1 is T_i and col2 is N_i
    graph.edge_index: (src=fiber_idx, tgt=class_idx)
    """

    # compute time prediction for each fiber-class edge
    time = gnn.edge_prediction(graph.x_e, scale=TOTAL_TIME/NCLASSES).squeeze(-1)

    # unpack
    src, tgt = graph.edge_index

    # compute class‐wise soft visit counts n_i'
    T_i = class_info[:, 0]  # required hours per visit for each class
    N_i = class_info[:, 1] / NFIELDS # total number of galaxies in each class, per field
    class_time = scatter(time, tgt, dim_size=NCLASSES, reduce='sum')
    n_prime = class_time / T_i # shape [NCLASSES]

    # soft rounding
    # n = softround(n_prime, sharpness, noiselevel)
    n = torch.round(n_prime)

    # class‐completeness = n_i / N_i
    completeness = n_prime / N_i
    # totutils = torch.min(completeness)
    totutils = torch.min(completeness)

    # penalty on per-class overallocation
    class_over = torch.relu(n_prime - N_i)
    class_penalty = pclass * torch.sum(class_over**2)

    # penalty on per‐fiber overtime
    fiber_time = scatter(time, src, dim_size=NFIBERS, reduce='sum')
    leaky = nn.LeakyReLU(negative_slope=0.1)
    overtime = fiber_time - TOTAL_TIME
    fiber_penalty = pfiber * torch.sum(leaky(overtime)**2)

    # final loss
    loss = -wutils * totutils + fiber_penalty + class_penalty

    if finaloutput:
        # optionally produce hard counts & diagnostics
        utils = torch.min(completeness)
        comp = (n / N_i).detach().cpu().numpy()
        fibers = fiber_time.detach().cpu().numpy()
        return loss, utils, comp, n, fibers 
    else:
        return loss, totutils

if __name__ == '__main__':
    # compute setup
    idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    ID = str(idx)

    # * loading class info
    class_info = torch.tensor(np.loadtxt('../' + datafile), dtype=torch.float, device=device)
    x_t = class_info
    # * fiber info: trivial counter so far
    x_s = torch.arange(NFIBERS, dtype=torch.float, device=device).reshape(-1, 1)

    # make a fully connected graph fibers -> classes
    edge_index = torch.cartesian_prod(torch.arange(NFIBERS), torch.arange(NCLASSES)).to(device).T

    # dummy inits for edges and globals
    x_e = torch.zeros(NFIBERS * NCLASSES,Fdim).to(device)
    x_u = torch.zeros(1, Fdim).to(device)

    # combine in graph
    graph = BipartiteData(edge_index=edge_index, x_s=x_s, x_t=x_t, x_e=x_e, x_u=x_u).to(device)

    # initialize model
    gnn = GNN(Fdim=Fdim, B=3, F_s=x_s.shape[1], F_t=x_t.shape[1], T=NCLASSES).to(device)
    gnn.train()

    # optimizers
    optimizer = torch.optim.Adam(gnn.parameters(), lr=lr)
    # stored for analysis 
    losses = np.zeros(nepochs)
    objective = np.zeros(nepochs)
    completions = np.zeros((NCLASSES, nepochs))
    fiber_time = np.zeros(NFIBERS)
    for epoch in trange(nepochs, desc=f'Training GNN ({str(device).upper()})'):
        # backprop
        gnn.zero_grad()
        graph_ = gnn(graph)
        loss, utility, completions[:,epoch], _, fiber_time = loss_function(graph_, class_info, pclass=pclass, pfiber=pfiber, finaloutput=True)
        # update parameters
        loss.backward()
        optimizer.step()
        # store for plotting
        losses[epoch] = loss.item()
        objective[epoch] = utility
    
    # print final results
    print(f'Final: Loss={losses[-1].item():.4e}, Utility={objective[-1]:.4f}')
    print(f'Completions: {completions[:,nepochs-1]}')

    # print theoretical optimum
    upper_bound = NFIBERS * TOTAL_TIME / torch.sum(torch.prod(class_info, dim=1)) * NFIELDS
    print(f'Upper Bound on Min Class Completion (Utility): {upper_bound}')
    
    now = datetime.now().strftime("%Y-%m-%d@%H-%M-%S")
    # torch.save(gnn.state_dict(), '../models/model_gnn_' + now + '.pth')

    # plot a histogram of the final fiber-time
    plt.figure(figsize=(6, 4))
    plt.hist(fiber_time, bins=30, color='blue', alpha=0.7)
    plt.axvline(x=TOTAL_TIME, color='red', linestyle='--', label='TOTAL_TIME')
    plt.xlabel('Fiber Time')
    plt.ylabel('Frequency')
    plt.title(rf'Final Fiber Time ($K = {fiber_time.shape[0]}$)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'../figures/B_{now}.png', dpi=600)
    
    # plot aggregate statistics
    epochs = np.arange(1, nepochs + 1)
    epochs_delayed = np.arange((start := 1 + max(nepochs - 100, 0)), nepochs + 1)
    plots_aggregate = [
        (epochs, losses, 'Epochs', 'Regularized Loss', 'red'),
        (epochs_delayed, losses[start-1:], 'Epochs', 'Regularized Loss', 'red'),
        (epochs, objective, 'Epochs', 'Min Class Completion', 'green')
    ]
    nrows = len(plots_aggregate)
    ncols = 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*4, nrows*3))
    fig.suptitle(rf'$F = {Fdim}$, $\eta = {lr}$, $N_{{e}} = {nepochs}$')
    for i, (xs, ys, xlabel, ylabel, color) in enumerate(plots_aggregate):
        ax = axes[i]
        ax.plot(xs, ys, color=color)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if i == 1: 
            ax.set_xlim(start, nepochs)
            step = max(1, (nepochs - start) // 5)
            ax.set_xticks(np.arange(start, nepochs+1, step))
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(fname=f'../figures/A_{now}.png', dpi=600)

    # plot per-class completion rates
    cmap = plt.get_cmap('viridis', NCLASSES)
    plots_class = []
    class_info = class_info.detach().cpu().numpy()
    for i in range(completions.shape[0]):
        plots_class.append(
            (epochs, completions[i], rf'Class {i+1} ($N_{{{i}}} = {int(class_info[i][1])}$)', cmap(i % cmap.N))
        )
    ncols = 2
    nrows = (NCLASSES + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*4, nrows*3)) # inches
    axes = axes.flatten()
    for idx, (xs, ys, title, color) in enumerate(plots_class):
        ax = axes[idx]
        ax.plot(xs, ys, color=color)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(1, nepochs)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    # remove any unused subplots
    for ax in axes[len(plots_class):]:
        fig.delaxes(ax)
    plt.tight_layout(rect=[0.05, 0.025, 0.95, 0.95])
    fig.supxlabel('Epochs')
    fig.supylabel('Completion')
    fig.suptitle(rf'$F = {Fdim}$, $\eta = {lr}$, $N_{{e}} = {nepochs}$')
    plt.savefig(f'../figures/C_{now}.png', dpi=600)
