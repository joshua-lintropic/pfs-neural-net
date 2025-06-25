import numpy as np
import torch
import torch.nn as nn
from torch_scatter import scatter
from gnn import GNN, BipartiteData
from params import *

import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import trange
from format import bcolors
import os

# === DEVICE SPEFICIATIONS ===
if torch.cuda.is_available():
    device = torch.device("cuda")
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
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

def loss_function(graph, class_info, penalty=1.0, sharpness=20, noiselevel=0.3, finaloutput=False):
    """
    time: [NCLASSES x NFIBERS], predicted t_{ik} for each fiber k, class i 
    properties: [NCLASSES, F_xt] where col1 is T_i and col2 is N_i
    graph.edge_index: (src=fiber_idx, tgt=class_idx)
    """

    # compute time prediction for each fiber-class edge
    time = gnn.edge_prediction(graph.x_e, scale=TOTAL_TIME/NCLASSES)

    # unpack
    src, tgt = graph.edge_index

    # compute class‐wise soft visit counts n_i'
    T_i = class_info[:, 0]  # required hours per visit for each class
    N_i = class_info[:, 1] / NFIELDS # total number of galaxies in each class, per field
    class_time = scatter(time.T, tgt, dim_size=graph.x_t.size(0), reduce='sum')
    n_prime = class_time / T_i # shape [NCLASSES]

    # soft rounding
    # n = softround(n_prime, sharpness, noiselevel)
    n = torch.round(n_prime)

    # class‐completeness = n_i / N_i
    completeness = n_prime / N_i
    totutils = torch.min(completeness)

    # penalty on per‐fiber overtime
    fiber_time = scatter(time.T, src, dim_size=graph.x_s.size(0), reduce='sum')
    overtime = fiber_time - TOTAL_TIME
    leaky = nn.LeakyReLU()  # squared‐leaky‐ReLU: p(x) = (LeakyReLU(x))^2
    penalty_term = penalty * torch.sum(leaky(overtime)**2)

    # final loss
    loss = -totutils + penalty_term

    if finaloutput:
        # optionally produce hard counts & diagnostics
        return loss, n / N_i, n, fiber_time
    else:
        return loss, totutils

if __name__ == '__main__':
    # compute setup
    idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    ID = str(idx)

    # * loading class info
    class_info = torch.tensor(np.loadtxt('../data/utils.txt'), dtype=torch.float, device=device)
    x_t = class_info
    # * fiber info: trivial counter so far
    x_s = torch.arange(NFIBERS, dtype=torch.float, device=device).reshape(-1, 1)

    # make a fully connected graph fibers -> classes
    edge_index = torch.cartesian_prod(torch.arange(NFIBERS), torch.arange(NCLASSES)).to(device).T

    # dummy inits for edges and globals
    F = 15 # lifted dimension
    x_e = torch.zeros(NFIBERS * NCLASSES,F)
    x_u = torch.zeros(1, F)

    # combine in graph
    graph = BipartiteData(edge_index=edge_index, x_s=x_s, x_t=x_t, x_e=x_e, x_u=x_u).to(device)

    # pre-training loop ---
    gnn = GNN(F=F, B=3, F_s=x_s.shape[1], F_t=x_t.shape[1], T=NCLASSES).to(device)
    # try:
    #     gnn.load_state_dict(torch.load('../models/model_gnn_pre' + ID + '.pth'))
    # except FileNotFoundError:
    #     print(f'{bcolors.WARNING}STATUS: No pre-trained checkpoint found.{bcolors.ENDC}')
    gnn.train()

    print(f'{bcolors.HEADER}STATUS: Start Pre-Training{bcolors.ENDC}')
    optimizer = torch.optim.Adam(gnn.parameters(), lr=1e-4)
    losses = np.zeros(nepochs)
    objective = np.zeros(nepochs)
    for epoch in trange(nepochs, desc='Training GNN'):
        # backprop
        gnn.zero_grad()
        graph_ = gnn(graph)
        loss, utility = loss_function(graph_, class_info, penalty=0.1)
        loss.backward()
        # print(f"Loss={loss.item():.4e}, Utility={utility:.4f}")
        optimizer.step()
        # store for plotting
        losses[epoch] = loss.item()
        objective[epoch] = utility
    torch.save(gnn.state_dict(), '../models/model_gnn_pre' + ID + '.pth')
    
    # plot results
    epochs = np.arange(1, nepochs + 1)
    epochs_delayed = np.arange((start := min(50, nepochs // 10)), nepochs + 1)
    plots = [
        (epochs, losses, 'Epochs', 'Regularized Loss', 'red'),
        (epochs_delayed, losses[start-1:], 'Epochs', 'Regularized Loss', 'blue'),
        (epochs, objective, 'Epochs', 'Min Class Completion', 'green')
    ]
    fig, axes = plt.subplots(nrows=len(plots))
    fig.suptitle(f'GNN Training Results for {nepochs} Epochs')
    for i, (xs, ys, xlabel, ylabel, color) in enumerate(plots):
        ax = axes[i]
        ax.plot(xs, ys, color=color)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if i != 1: continue
        ax.set_xlim(start, nepochs)
        step = max(1, (nepochs - start) // 5)
        ax.set_xticks(np.arange(start, nepochs+1, step))
    plt.tight_layout()
    figname = datetime.now().strftime("%Y-%m-%d@%H:%M:%S")
    plt.savefig(fname=f'../figures/{figname}.png', dpi=600)
