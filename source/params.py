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
nepochs = 10000
Fdim = 15 # lifted dimension