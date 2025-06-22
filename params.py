import torch

# === FEATURE DIMENSIONS === 
F_e = 10       # number of features per edge
F_u = 10       # number of features for global node
F_xs = 10      # number of features for source node (fibers)
F_xt = 2       # number of features for target node (classes)
F_e_out = 5    # number of field IDs for output

# === CONSTANTS ===
NFIBERS = 2000  # number of fibers
NCLASSES = 12   # number of classes
TOTAL_TIME = 6  # per‑fiber time budget (T)

# === DEVICE SPECS ===
device = torch.device('mps')   # 'cuda' (gpu), 'mps' (apple silicon), 'cpu' (other)
