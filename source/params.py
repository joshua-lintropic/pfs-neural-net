import torch

# === DEVICE SPECS ===
device = torch.device('mps')   # 'cuda' (gpu), 'mps' (apple silicon), 'cpu' (other)

# === FEATURE DIMENSIONS === 
F_e = 10       # number of features per edge
F_u = 10       # number of features for global node
F_xs = 10      # number of features for source node (fibers)
F_xt = 2       # number of features for target node (classes)
F_e_out = 5    # number of field IDs for output

# === CONSTANTS ===
NFIBERS = 2000  # number of fibers
NCLASSES = 12   # number of classes
TOTAL_TIME = 8  # per‑fiber time budget (T)

# === HYPERPARAMETERS === 
ntrain, ntest = 1, 1
batchsize = 1
sharpness = 20
noiselevel = 0.2
nepoch_pre, nepoch = 1, 10
lr_pre, lr = 5e-4, 1e-4
penalty_pre = 1e-1
penalty_ini, penalty_end = 1.0, 10.0
train = True
batchsize = min(batchsize, ntrain)