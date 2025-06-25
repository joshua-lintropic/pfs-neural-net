import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# === CONSTANTS ===
NFIBERS = 2000  # number of fibers
NCLASSES = 12   # number of classes
NFIELDS = 10    # number of GE fields
TOTAL_TIME = 40 # per‑fiber time budget

# === TRAINING PARAMETERS ===
nepochs = 1000
Fdim = 10 # lifted dimension
lr = 1e-3 # learning rate
pclass = 0.1
pfiber = 0.1