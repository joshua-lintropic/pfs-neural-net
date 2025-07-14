import numpy as np
import torch
import torch.nn as nn
from torch_scatter import scatter
from gnn import GNN, BipartiteData
from config import *

import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import trange
import os
import sys

# === DEVICE SPEFICIATIONS ===
ncores = os.cpu_count() or 1
os.environ['OMP_NUM_THREADS'] = str(ncores)
os.environ['MKL_NUM_THREADS'] = str(ncores)
torch.set_num_threads(ncores)
torch.set_num_interop_threads(ncores)

def softfloor(x, sharpness=20, noiselevel=0.3):
    noise = noiselevel * (torch.rand_like(x) - 0.5)
    x = x + noise
    sharpness = x.new_tensor(sharpness)
    pi = x.new_tensor(np.pi)
    r = torch.where(sharpness == 0, torch.tensor(0.0, device=x.device), torch.exp(-1/sharpness))
    return x + 1 / pi * (torch.arctan(r * torch.sin(2 * pi * x) / (1 - r * torch.cos(2 * pi * x))) - torch.arctan(r / (torch.ones_like(r) - r)))

def loss_function(graph, class_info, pclass=0.1, pfiber=1.0, sharpness=0.5, finaloutput=False):
    """
    time: [NCLASSES x NFIBERS], predicted t_{ik} for each fiber k, class i 
    properties: [NCLASSES, F_xt] where col1 is T_i and col2 is N_i
    graph.edge_index: (src=fiber_idx, tgt=class_idx)
    """
    # unpack
    src, tgt = graph.edge_index

    # compute class‐wise soft visit counts 
    T_i = class_info[:, 0]  # required hours per visit for each class
    T_i = T_i.unsqueeze(0).expand(NFIBERS, -1).reshape(-1)
    N_i = class_info[:, 1] / NFIELDS # total number of galaxies in each class, per field
    time = gnn.edge_prediction(graph.x_e, scale=TOTAL_TIME/NCLASSES).squeeze(-1)
    visited = time / T_i

    # compute number of observed galaxies
    galaxies = softfloor(visited, sharpness)
    galaxies = torch.maximum(torch.full_like(galaxies, 0.0), galaxies)
    n_prime = scatter(galaxies, tgt, dim_size=NCLASSES, reduce='sum')
    time = galaxies * T_i
    # n = scatter(torch.floor(visited), tgt, dim_size=NCLASSES, reduce='sum')

    # class‐completeness = n_i / N_i
    completeness = n_prime / N_i
    totutils = torch.min(completeness)

    # penalty on per-class overallocation
    class_over = torch.relu(n_prime - N_i)
    class_penalty = pclass * torch.sum(class_over**2)

    # penalty on per‐fiber overtime
    fiber_time = scatter(time, src, dim_size=NFIBERS, reduce='sum')
    overtime = fiber_time - TOTAL_TIME
    leaky = nn.LeakyReLU(negative_slope=0.1)
    fiber_penalty = pfiber * torch.sum(leaky(overtime)**2)

    # encourage variance 
    Time = time.reshape(NFIBERS, NCLASSES)
    variance = torch.sum(torch.var(Time, dim=0))

    # final loss
    loss = -wutils * totutils + fiber_penalty + class_penalty - wvar * variance

    if finaloutput:
        # optionally produce hard counts & diagnostics
        utils = torch.min(completeness)
        comp = (n_prime / N_i).detach().cpu().numpy()
        fibers = fiber_time.detach().cpu().numpy()
        return loss, utils, comp, n_prime, fibers, time, variance
    else:
        return loss, totutils

if __name__ == '__main__':
    # compute setup
    idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    ID = str(idx)

    # * loading class info
    class_info = torch.tensor(np.loadtxt(datafile), dtype=torch.float, device=device)
    x_t = class_info
    # * fiber info: trivial counter so far
    x_s = torch.arange(NFIBERS, dtype=torch.float, device=device).reshape(-1, 1)

    # make a fully connected graph fibers -> classes
    edge_index = torch.cartesian_prod(torch.arange(NFIBERS), torch.arange(NCLASSES)).to(device).T

    # dummy inits for edges and globals
    lo = 2.0
    hi = 10.0
    # x_e = torch.zeros(NFIBERS * NCLASSES,Fdim).to(device)
    x_e = lo + (hi - lo) * torch.rand(size=(NFIBERS * NCLASSES, Fdim)).to(device)
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
    best_utility = 0.0
    best_loss = 0.0
    best_time = np.zeros(NCLASSES * NFIBERS)
    best_fiber_time = np.zeros(NFIBERS)
    best_completion = np.zeros(NCLASSES)
    variances = np.zeros(nepochs)

    # training loop
    start_epoch = 0
    model_path = sys.argv[1] if len(sys.argv) > 1 else ""
    if model_path:
        checkpoint = torch.load(model_path, map_location=device)
        gnn.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optim_state'])
        start_epoch = checkpoint['epoch'] + 1
    for epoch in trange(start_epoch, nepochs, desc=f'Training GNN ({str(device).upper()})'):
        # backprop
        gnn.zero_grad()
        graph_ = gnn(graph)
        sharp = sharps[0] + (sharps[1] - sharps[0]) * epoch / nepochs
        loss, utility, completions[:,epoch], _, fiber_time, time, variance = loss_function(graph_, class_info, pclass=pclass, pfiber=pfiber, sharpness=sharp, finaloutput=True)
        # update parameters
        loss.backward()
        optimizer.step()
        # store for plotting
        losses[epoch] = loss.item()
        objective[epoch] = utility
        variances[epoch] = variance
        if utility > best_utility and sharp > min_sharp: 
            # update bests
            best_loss = loss.item()
            best_utility = utility
            best_time = time
            best_fiber_time = fiber_time
            best_completion = completions[:,epoch]
            # checkpoint the model
            torch.save({
                'epoch': epoch, 
                'model_state': gnn.state_dict(),
                'optim_state': optimizer.state_dict()
            }, checkpoint_path + ID + '.pth')

    # checkpoint the final model
    torch.save({
        'epoch': nepochs, 
        'model_state': gnn.state_dict(),
        'optim_state': optimizer.state_dict()
    }, checkpoint_path + ID + '.pth')

    
    # write final results to output log
    now = datetime.now().strftime("%Y-%m-%d@%H-%M-%S")
    upper_bound = NFIBERS * TOTAL_TIME / torch.sum(torch.prod(class_info, dim=1)) * NFIELDS
    with open('../figures/L_' + now + '.txt', 'w') as f:
        f.write(f'TIMESTAMP: {now}\n')
        f.write(f'Best: Loss={best_loss:.4e}, Utility={best_utility:.4f}\n')
        f.write(f'Best Completion: {best_completion]}\n')
        f.write(f'Upper Bound on Min Class Completion (Utility): {upper_bound}\n')

    # === PLOT FINAL FIBER-TIME HISTOGRAM === #
    plt.figure(figsize=(6, 4))
    plt.hist(best_fiber_time, bins=30, color='blue', alpha=0.7)
    plt.axvline(x=TOTAL_TIME, color='red', linestyle='--', label='TOTAL_TIME')
    plt.xlabel('Fiber Time')
    plt.ylabel('Frequency')
    plt.title(rf'Final Fiber Time ($K = {best_fiber_time.shape[0]}$)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'../figures/B_{now}.png', dpi=600)
    
    # === PLOT AGGREGATE STATISTICS ===
    epochs = np.arange(1, nepochs + 1)
    epochs_delayed = np.arange((start := 1 + max(nepochs - 100, 0)), nepochs + 1)
    plots_aggregate = [
        (epochs, losses, 'Epochs', 'Regularized Loss', 'red'),
        (epochs_delayed, losses[start-1:], 'Epochs', 'Regularized Loss', 'red'),
        (epochs, objective, 'Epochs', 'Min Class Completion', 'green'),
        (epochs, variances, 'Epochs', 'Variance', 'blue')
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
        if i == 2:
            ax.axhline(y=upper_bound.detach().cpu().numpy(), color='blue')
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(fname=f'../figures/A_{now}.png', dpi=600)

    # === PLOT PER-CLASS COMPLETION RATES === #
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

    # === PLOT FIBER ACTIONS FOR RANDOM FIBERS === #
    fibers_rand = np.random.randint(low=0, high=NFIBERS, size=(10,))
    fibers_slice = np.array(list(range(5)) + list(range(NFIBERS-5,NFIBERS)))

    class_req = class_info[:, 0]

    def plot_fiber_actions(fibers, char):
        # build a dict of raw times for each fiber
        dist = {k: best_time[k * NCLASSES:(k + 1) * NCLASSES].detach().cpu().numpy() for k in fibers}

        # stack into 2D array: rows = fibers, cols = classes
        raw_data = np.vstack([dist[k] for k in fibers])

        # 1) Round up each allocation to nearest multiple of time requirement
        rounded = np.round(raw_data / class_req) * class_req

        # compute offsets for stacked bar
        cumulative = np.cumsum(rounded, axis=1)
        left = np.hstack([np.zeros((rounded.shape[0], 1)), cumulative[:, :-1]])

        # prepare plot
        fig, ax = plt.subplots(figsize=(8, 6))
        y = np.arange(len(fibers))
        height = 0.8

        cmap = plt.get_cmap('viridis', NCLASSES)
        for cls in range(NCLASSES):
            ax.barh(
                y,
                rounded[:, cls],
                left=left[:, cls],
                height=height,
                color=cmap(cls),
                edgecolor='none',
                label=f'Class {cls + 1}'
            )

            # 2) demarcate individual targets with lines
            for i, fiber in enumerate(fibers):
                n_targets = round(rounded[i, cls] / class_req[cls])
                for m in range(1, n_targets):
                    x = left[i, cls] + m * class_req[cls]
                    ax.vlines(
                        x,
                        y[i] - height/2,
                        y[i] + height/2,
                        colors='white',
                        linestyles='--',
                        linewidth=0.8
                    )

        # formatting
        ax.set_yticks(y)
        ax.set_yticklabels(fibers)
        ax.invert_yaxis()
        ax.set_xlabel('Time (hours)')
        ax.set_title('Fiber Class-Times (rounded & segmented)')
        ax.legend(loc='best', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig(f'../figures/{char}_{now}.png', dpi=600)

    plot_fiber_actions(fibers_rand, 'D')
    plot_fiber_actions(fibers_slice, 'E')
