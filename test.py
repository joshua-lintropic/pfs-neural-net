# galaxy_assignment_gnn.py
# PyTorch ≥1.13

from typing import List, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------  data helpers  -----------------------------

class GalaxyInstance:
    """
    Holds one problem instance:
        K   – number of fibers (int, 2000)
        L   – number of exposures (int, 10)
        class_of[i]        – int in [0,M-1]
        T_of_class[m]      – tensor [M] per-galaxy requirement
        O_of_galaxy[i]     – tensor [N] already-observed counts
        fiber_neighbors[k] – list[int] of galaxy indices reachable by fiber k
                             (same set used for all exposures)
    """
    def __init__(self,
                 fiber_neighbors: List[List[int]],
                 class_of: torch.LongTensor,
                 T_of_class: torch.Tensor,
                 O_of_galaxy: torch.Tensor):
        self.K  = len(fiber_neighbors)
        self.L  = 10
        self.N  = len(class_of)
        self.M  = int(T_of_class.numel())

        self.fiber_neighbors = fiber_neighbors
        self.class_of        = class_of          # [N]
        self.T_of_class      = T_of_class        # [M]
        self.O_of_galaxy     = O_of_galaxy       # [N]

        # pre-compute remaining need
        D = T_of_class[class_of] - O_of_galaxy   # broadcast
        self.D_of_galaxy = torch.clamp(D, min=0) # [N]

        # galaxies already done -> mask
        self.active_mask  = self.D_of_galaxy > 0  # bool [N]

        # remove completed galaxies from each fiber's reachability
        self.active_neighbors = [
            [g for g in nbrs if self.active_mask[g]] for nbrs in fiber_neighbors
        ]


# ---------------------------  graph utilities  ---------------------------

def make_fiber_exposure_features(K:int, L:int, device):
    """Return tensor [K*L, 2] with (fiber_idx, exposure_idx)."""
    grid_f, grid_e = torch.meshgrid(
        torch.arange(K, device=device),
        torch.arange(L, device=device),
        indexing="ij"
    )
    feats = torch.stack([grid_f.flatten(), grid_e.flatten()], dim=1).float()
    return feats                                           # shape [K*L, 2]

def aggregate_mean(src, index, dim_size):
    """
    src  : tensor [E, dim]
    index: tensor [E] destination node indices
    Returns mean aggregated tensor [dim_size, dim]
    """
    out = torch.zeros(dim_size, src.size(1), device=src.device)
    out.scatter_add_(0, index.unsqueeze(-1).expand_as(src), src)
    counts = torch.bincount(index, minlength=dim_size).clamp(min=1).unsqueeze(-1)
    return out / counts


# ---------------------------  the GNN module  ----------------------------

class GalaxyAssignmentGNN(nn.Module):
    def __init__(self,
                 embed_dim: int = 128,
                 msg_layers: int = 4,
                 use_class_nodes: bool = True,
                 temperature: float = 1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.Layers    = msg_layers
        self.temp      = temperature
        self.use_class = use_class_nodes

        # -- input embeddings (tiny MLPs) --
        self.fiber_in  = nn.Linear(2, embed_dim)      # (fiber idx, exposure idx)
        self.gal_in    = nn.Linear(4, embed_dim)      # (class id, T, O, D)
        if use_class_nodes:
            self.cls_in = nn.Linear(2, embed_dim)     # (N_m, T_m)

        # -- message-passing weights --
        self.fiber_upd  = nn.ModuleList(
            nn.Linear(embed_dim*2, embed_dim) for _ in range(msg_layers)
        )
        self.gal_upd    = nn.ModuleList(
            nn.Linear(embed_dim*2, embed_dim) for _ in range(msg_layers)
        )
        if use_class_nodes:
            self.cls_upd = nn.ModuleList(
                nn.Linear(embed_dim*2, embed_dim) for _ in range(msg_layers)
            )

        # -- edge scorer --
        self.scorer = nn.Linear(embed_dim*2, 1)

    # ----------  forward ----------
    def forward(self, inst: GalaxyInstance):
        """
        Returns:
            assign_probs – list with length K*L;
                           each element is (gal_idx_tensor, prob_tensor)
                           where prob sums to 1 over that fiber-exposure’s choices.
        """
        dev = next(self.parameters()).device
        K, L, N, M = inst.K, inst.L, inst.N, inst.M
        KL = K*L

        # -- node initial embeddings --
        f_feats = make_fiber_exposure_features(K, L, dev)    # [KL, 2]
        f_emb   = F.relu(self.fiber_in(f_feats))             # [KL, d]

        gal_feats_raw = torch.stack([
            inst.class_of.float(),
            inst.T_of_class[inst.class_of],  # per galaxy T_m
            inst.O_of_galaxy,
            inst.D_of_galaxy
        ], dim=1).to(dev)                                     # [N,4]
        g_emb = F.relu(self.gal_in(gal_feats_raw))            # [N, d]

        if self.use_class:
            # gather N_m for each class
            Nm = torch.bincount(inst.class_of, minlength=M).float().to(dev)
            cls_feats_raw = torch.stack([Nm, inst.T_of_class], dim=1)  # [M,2]
            c_emb = F.relu(self.cls_in(cls_feats_raw))

        # -- pre-build index tensors for aggregation --
        # edge list: we need indices in flat KL space
        fiber_id   = []
        gal_id     = []
        for k in range(K):
            nbrs = inst.active_neighbors[k]  # already filtered
            if not nbrs:
                continue
            for l in range(L):
                idx_f = k*L + l
                fiber_id.extend([idx_f]*len(nbrs))
                gal_id.extend(nbrs)
        fiber_id = torch.tensor(fiber_id, device=dev, dtype=torch.long)
        gal_id   = torch.tensor(gal_id,   device=dev, dtype=torch.long)

        # message-passing
        for layer in range(self.Layers):
            # fiber -> galaxy mean
            msg_f2g = f_emb[fiber_id]                          # [E,d]
            g_aggr  = aggregate_mean(msg_f2g, gal_id, N)       # [N,d]

            g_emb = F.relu(self.gal_upd[layer](
                torch.cat([g_emb, g_aggr], dim=-1)))

            # (optional) galaxy -> class -> galaxy path
            if self.use_class:
                gal_cls = inst.class_of.to(dev)                # [N]
                msg_g2c = g_emb                                 # [N,d]
                c_aggr  = aggregate_mean(msg_g2c, gal_cls, M)   # [M,d]

                c_emb = F.relu(self.cls_upd[layer](
                    torch.cat([c_emb, c_aggr], dim=-1)))

                # broadcast back
                g_emb = g_emb + c_emb[gal_cls]

            # galaxy -> fiber mean
            msg_g2f = g_emb[gal_id]                            # [E,d]
            f_aggr  = aggregate_mean(msg_g2f, fiber_id, KL)    # [KL,d]

            f_emb = F.relu(self.fiber_upd[layer](
                torch.cat([f_emb, f_aggr], dim=-1)))

        # ---------  edge scoring & softmax  ---------
        pair_emb = torch.cat(
            [f_emb[fiber_id], g_emb[gal_id]], dim=-1)          # [E,2d]
        raw_score = self.scorer(pair_emb).squeeze(-1) / self.temp

        # per fiber-exposure softmax
        probs = torch.zeros_like(raw_score)
        # walk edges in original order per fiber
        start = 0
        assign_probs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for k in range(K):
            nbrs = inst.active_neighbors[k]
            if not nbrs:
                for l in range(L):
                    assign_probs.append((torch.empty(0,dtype=torch.long,device=dev),
                                         torch.empty(0, device=dev)))
                continue
            for l in range(L):
                e = len(nbrs)
                sl = raw_score[start : start+e]
                pl = F.softmax(sl, dim=0)
                probs[start : start+e] = pl
                start += e
                assign_probs.append((torch.tensor(nbrs, device=dev), pl))

        return assign_probs, (f_emb, g_emb)

    # -------------  utility: build P[l,k,i] tensor  -------------
    @staticmethod
    def assignment_matrix(assign_probs: List[Tuple[torch.Tensor, torch.Tensor]],
                          inst: GalaxyInstance) -> torch.Tensor:
        """Return float tensor [L,K,N] with probabilities."""
        device = assign_probs[0][1].device
        P = torch.zeros(inst.L, inst.K, inst.N, device=device)
        idx = 0
        for k in range(inst.K):
            for l in range(inst.L):
                gal_idx, prob = assign_probs[idx]
                if prob.numel() > 0:
                    P[l, k, gal_idx] = prob
                idx += 1
        return P  # [L,K,N]


# ------------------------  loss / objective  ------------------------

def compute_loss(model: GalaxyAssignmentGNN,
                 inst : GalaxyInstance,
                 assign_probs: List[Tuple[torch.Tensor, torch.Tensor]],
                 eta1: float = 500.0,
                 eta2: float = 500.0,
                 beta: float = 50.0,
                 lambda_param: float = 1.0) -> torch.Tensor:

    P = model.assignment_matrix(assign_probs, inst)   # [L,K,N]
    dev = P.device

    # total new observations per galaxy
    A = P.sum(dim=(0,1))                              # [N]

    D = inst.D_of_galaxy.to(dev)                      # [N]

    # --- class fairness scores ---
    class_ids = inst.class_of.to(dev)
    M = inst.M
    f_m = []
    for m in range(M):
        mask = (class_ids == m) & (D > 0)
        if mask.sum() == 0:
            # class already fully satisfied; score 1
            f_m.append(torch.tensor(1.0, device=dev))
            continue
        avg_prog = ((A - D)[mask]).mean() / lambda_param
        s = torch.sigmoid(avg_prog)
        f_m.append(s)
    f = torch.stack(f_m)                              # [M]

    # smooth minimum (soft-min)
    soft_min = -torch.log(torch.exp(-beta*f).mean()) / beta

    # --- constraints ---
    # ≤1 galaxy per fiber per exposure is automatic (softmax), but we still
    # penalise >1 galaxy per galaxy per exposure
    gal_over_exposure = torch.clamp(P.sum(dim=1) - 1.0, min=0)  # [L,N]
    penalty1 = (gal_over_exposure**2).sum()

    # total overshoot per galaxy
    gal_over_total = torch.clamp(A - D, min=0)
    penalty2 = (gal_over_total**2).sum()

    return -soft_min + eta1*penalty1 + eta2*penalty2


# ------------------------  tiny training stub  ------------------------

def train_one_epoch(model: GalaxyAssignmentGNN,
                    instances: List[GalaxyInstance],
                    opt: torch.optim.Optimizer):
    model.train()
    total_loss = 0.0
    for inst in instances:
        opt.zero_grad()
        assign_probs, _ = model(inst)
        loss = compute_loss(model, inst, assign_probs)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    return total_loss / len(instances)


# ---------------------------------------------------------------------
if __name__ == "__main__":
    # quick dummy demo to show the script runs
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # fabricate a *tiny* random instance (K=8, N=30, M=3) just to sanity-check
    K, L, N, M = 8, 10, 30, 3
    fiber_neighbors = [torch.randperm(N).tolist() for _ in range(K)]  # all 30 reachable
    class_of = torch.randint(0, M, (N,))
    T_of_class = torch.randint(2, 5, (M,)).float()
    O_of_galaxy = torch.randint(0, 2, (N,)).float()  # some already observed

    inst = GalaxyInstance(fiber_neighbors, class_of, T_of_class, O_of_galaxy)

    model = GalaxyAssignmentGNN(embed_dim=64, msg_layers=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    inst_gpu = GalaxyInstance(fiber_neighbors, class_of.cuda(), T_of_class.cuda(), O_of_galaxy.cuda())
    
    num_epochs = 10  # set number of training epochs
    for epoch in range(num_epochs):
        loss_epoch = train_one_epoch(model, [inst_gpu], opt)
        print(f"Epoch {epoch+1}/{num_epochs}: training loss = {loss_epoch:.4f}")
    
    # Final loss after training
    loss0 = train_one_epoch(model, [inst_gpu], opt)
    print("Final training loss:", loss0)
