import torch
import torch.nn.functional as F
import torch_geometric
from torch_scatter import scatter, scatter_mean
from params import *

class Argmax(torch.autograd.Function):
    """
    Custom autograd Function that computes a one-hot argmax in the forward pass
    and uses a finite-difference straight-through estimator in the backward pass.
    """

    @staticmethod
    def forward(context, input):
        """
        Forward pass: computes the one-hot encoding of the maximum value indices.

        Args:
            context: Context object to save tensors for backward.
            input (Tensor): Input tensor of shape [..., F_e_out].

        Returns:
            Tensor: One-hot encoded tensor of the same shape as input, with
            a 1 at the index of the maximum value in the last dimension.
        """
        _, indices = torch.max(input, dim=1)
        result = F.one_hot(indices, num_classes=F_e_out).float()
        context.save_for_backward(input, result)
        return result

    @staticmethod
    def backward(context, grad_output):
        """
        Backward pass: approximates gradients using a straight-through estimator.

        Args:
            context: Context object with saved tensors from forward.
            grad_output (Tensor): Gradient of the loss with respect to the output.

        Returns:
            Tensor: Gradient with respect to the input of the forward pass.
        """
        Lambda = 1000
        input, result = context.saved_tensors
        
        # NOTE: perturb input by scaled upstream gradient
        input1 = input + Lambda * grad_output
        _, indices1 = torch.max(input1, dim=-1)
        result1 = F.one_hot(indices1, num_classes=F_e_out)

        # NOTE: finite-difference approximation of the gradient
        grad = - (result - result1) / Lambda
        return grad

def sparse_sort(src, index, dim=0, descending=False, eps=1e-12):
    """
    Computes a stable sorting permutation of `src` along a given dimension,
    using `index` as a tie-breaker.

    Args:
        src (Tensor): Values to sort.
        index (Tensor): Integer tensor of same shape as src, used to break ties.
        dim (int): Dimension along which to sort.
        descending (bool): If True, sort in descending order.
        eps (float): Small constant to avoid division by zero.

    Returns:
        LongTensor: Permutation indices that sort `src` in the desired order.
    """
    f_src = src.float()
    f_min, f_max = f_src.min(dim)[0], f_src.max(dim)[0]
    norm = (f_src - f_min) / (f_max - f_min + eps) + index.float() * (-1)**int(descending)
    perm = norm.argsort(dim=dim, descending=descending)
    return perm

class BipartiteData(torch_geometric.data.Data):
    """
    Data class for a bipartite graph with separate source and target node sets.

    Attributes:
        x_s (Tensor): Source-node features [N_s, F_xs].
        x_t (Tensor): Target-node features [N_t, F_xt].
        edge_index (LongTensor): Edge indices [2, E], row0=src, row1=tgt.
        edge_attr (Tensor): Edge features [E, F_e].
        u (Tensor): Global features [B, F_u].
    """
    def __init__(self, edge_index, x_s, x_t, edge_attr, u):
        # super().__init__(edge_index=edge_index, x_s=x_s, x_t=x_t, edge_attr=edge_attr, u=u)
        super(BipartiteData, self).__init__()
        if edge_index is not None:
            self.edge_index = edge_index.to(device)
        if x_s is not None:
            self.x_s = x_s.to(device)
        if x_t is not None:
            self.x_t = x_t.to(device)
            self.num_nodes = len(self.x_t)  # dummy to suppress warnings
        if edge_attr is not None:
            self.edge_attr = edge_attr.to(device)
        if u is not None:
            self.u = u.to(device)

    def __inc__(self, key, value, *args):
        """
        Defines how to increment indices when batching multiple graphs.

        Args:
            key (str): Name of the attribute to increment.
            value: The existing value of that attribute.

        Returns:
            Tensor: The increment size for edge_index; otherwise delegates to super().
        """
        if key == 'edge_index':
            # Shift sources by N_s, targets by N_t when batching
            return torch.tensor([[self.x_s.size(0)], [self.x_t.size(0)]]).to(device)
        else:
            return super().__inc__(key, value, *args)

class Loader(torch_geometric.data.Dataset):
    """
    Simple Dataset wrapping a list of BipartiteData graphs.
    """
    def __init__(self, graphs_list=None):
        """Initialize with a list of Data objects."""
        self.graphs_list = graphs_list

    def __len__(self):
        """Return number of graphs."""
        return len(self.graphs_list)

    def __getitem__(self, idx):
        """Retrieve graph at index idx."""
        return self.graphs_list[idx]


class EdgeModel(torch.nn.Module):
    """
    Edge update module: combines source/node features, target/node features,
    edge features, and global features into updated edge embeddings.
    """
    def __init__(self):
        super(EdgeModel, self).__init__()
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(F_xs + F_xt + F_e + F_u, F_e),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(F_e, F_e)
        )

    def forward(self, x_s, x_t, edge_index, edge_attr, u, batch_e):
        """
        Args:
            x_s (Tensor): Source-node embeddings [E_s, F_xs].
            x_t (Tensor): Target-node embeddings [E_t, F_xt].
            edge_index (LongTensor): [2, E].
            edge_attr (Tensor): Edge embeddings [E, F_e].
            u (Tensor): Global embeddings [B, F_u].
            batch_e (LongTensor): Edge-to-graph map [E].

        Returns:
            Tensor: Updated edge features [E, F_e].
        """
        src, tgt = edge_index

        # print("\nEDGE MODEL FORWARD")
        # print(f"src ({src.shape}):", src)
        # print(f"tgt ({tgt.shape}):", tgt)
        # print(f"x_s ({x_s.shape}):", x_s)
        # print(f"x_t ({x_t.shape}):", x_t)
        # print(f"edge_attr ({edge_attr.shape}):", edge_attr)
        # print(f"u ({u.shape}):", u)
        
        h = torch.cat([x_s[src], x_t[tgt], edge_attr, u[batch_e]], dim=-1)
        # print(f"h ({h.shape})")
        # print(f"x_s[src] ({x_s[src].shape})")
        # print(f"x_t[tgt] ({x_t[tgt].shape})")
        # print(f"edge_attr ({edge_attr.shape})")
        # print(f"u[batch_e] {(u[batch_e].shape)}")
        return self.edge_mlp(h)


class EdgeModel_out(torch.nn.Module):
    """
    Final edge update module producing logits over discrete edge labels.
    """
    def __init__(self):
        super(EdgeModel_out, self).__init__()
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(F_xs + F_xt + F_e + F_u, F_e_out),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(F_e_out, F_e_out)
        )

    def forward(self, x_s, x_t, edge_index, edge_attr, u, batch_e):
        """
        Same as EdgeModel but outputs dimension F_e_out.
        """
        src, tgt = edge_index
        h = torch.cat([x_s[src], x_t[tgt], edge_attr, u[batch_e]], dim=-1)
        return self.edge_mlp(h)


class SModel(torch.nn.Module):
    """
    Source-node update: aggregates incoming edge messages (with statistics)
    and updates source-node embeddings.
    """
    def __init__(self):
        super(SModel, self).__init__()
        self.node_mlp_1 = torch.nn.Sequential(
            torch.nn.Linear(F_e + F_xt, F_e + F_xt),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(F_e + F_xt, F_e + F_xt)
        )
        self.node_mlp_2 = torch.nn.Sequential(
            torch.nn.Linear(F_xs + 1 + 4 * (F_e + F_xt) + F_u, F_xs),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(F_xs, F_xs)
        )

    def forward(self, x_s, x_t, edge_index, edge_attr, u, batch_s):
        """
        Args:
            x_s (Tensor): Source embeddings [N_s, F_xs].
            x_t (Tensor): Target embeddings [N_t, F_xt].
            edge_index (LongTensor): [2, E].
            edge_attr (Tensor): Edge embeddings [E, F_e].
            u (Tensor): Global embeddings [B, F_u].
            batch_s (LongTensor): Node-to-graph map for source nodes [N_s].

        Returns:
            Tensor: Updated source-node features [N_s, F_xs].
        """
        src, tgt = edge_index
        msg = torch.cat([x_t[tgt], edge_attr], dim=1)
        msg = self.node_mlp_1(msg)

        # Count and aggregate stats of incoming messages
        count = scatter(torch.ones(len(msg), 1, device=device), src, dim=0,
                        dim_size=x_s.size(0), reduce='sum')
        mean = scatter(msg, src, dim=0, dim_size=x_s.size(0), reduce='mean')
        var = F.relu(scatter(msg**2, src, dim=0, dim_size=x_s.size(0), reduce='mean') - mean**2)
        std = torch.sqrt(var + 1e-6)
        skew = scatter((msg - mean[src])**3, src, dim=0, dim_size=x_s.size(0), reduce='mean') / std**3
        kurt = scatter((msg - mean[src])**4, src, dim=0, dim_size=x_s.size(0), reduce='mean') / std**4

        # zero out if nan
        mean = torch.nan_to_num(mean, nan=0.0)
        var  = torch.nan_to_num(var,  nan=0.0)
        std  = torch.sqrt(var + 1e-6)
        skew = torch.nan_to_num(skew, nan=0.0)
        kurt = torch.nan_to_num(kurt, nan=0.0)

        h_cat = torch.cat([x_s, count, mean, std, skew, kurt, u[batch_s]], dim=1)
        return self.node_mlp_2(h_cat)


class TModel(torch.nn.Module):
    """
    Target-node update: sums incoming edge messages and updates target-node embeddings.
    """
    def __init__(self):
        super(TModel, self).__init__()
        self.node_mlp_1 = torch.nn.Sequential(
            torch.nn.Linear(F_e + F_xs, F_e + F_xs),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(F_e + F_xs, F_e + F_xs)
        )
        self.node_mlp_2 = torch.nn.Sequential(
            torch.nn.Linear(F_xt + (F_e + F_xs) + F_u, F_xt),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(F_xt, F_xt)
        )

    def forward(self, x_s, x_t, edge_index, edge_attr, u, batch_t):
        """
        Args:
            x_s (Tensor): Source embeddings [N_s, F_xs].
            x_t (Tensor): Target embeddings [N_t, F_xt].
            edge_index (LongTensor): [2, E].
            edge_attr (Tensor): Edge embeddings [E, F_e].
            u (Tensor): Global embeddings [B, F_u].
            batch_t (LongTensor): Node-to-graph map for target nodes [N_t].

        Returns:
            Tensor: Updated target-node features [N_t, F_xt].
        """
        src, tgt = edge_index
        msg = torch.cat([x_s[src], edge_attr], dim=1)
        msg = self.node_mlp_1(msg)
        agg = scatter(msg, tgt, dim=0, dim_size=x_t.size(0), reduce='sum')
        h_cat = torch.cat([x_t, agg, u[batch_t]], dim=1)
        return self.node_mlp_2(h_cat)


class GlobalModel(torch.nn.Module):
    """
    Graph-level update: pools node embeddings to update global features.
    """
    def __init__(self):
        super(GlobalModel, self).__init__()
        self.global_mlp = torch.nn.Sequential(
            torch.nn.Linear(F_u + F_xs + F_xt, F_u),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(F_u, F_u)
        )

    def forward(self, x_s, x_t, edge_index, edge_attr, u, batch_s, batch_t):
        """
        Args:
            x_s (Tensor): Updated source-node features [N_s, F_xs].
            x_t (Tensor): Updated target-node features [N_t, F_xt].
            edge_index (LongTensor): [2, E].
            edge_attr (Tensor): Updated edge features [E, F_e].
            u (Tensor): Previous global features [B, F_u].
            batch_s (LongTensor): Source-node graph indices [N_s].
            batch_t (LongTensor): Target-node graph indices [N_t].

        Returns:
            Tensor: Updated global features [B, F_u].
        """
        s_mean = scatter_mean(x_s, batch_s, dim=0)
        t_mean = scatter_mean(x_t, batch_t, dim=0)
        h_cat = torch.cat([u, s_mean, t_mean], dim=1)
        return self.global_mlp(h_cat)


class Block(torch.nn.Module):
    """
    A single MetaLayer block combining edge, source-node, target-node,
    and global update modules.
    """
    def __init__(self, edge_model=None, s_model=None, t_model=None, u_model=None):
        super(Block, self).__init__()
        self.edge_model   = edge_model
        self.s_model      = s_model
        self.t_model      = t_model
        self.global_model = u_model

    def forward(self, x_s, x_t, edge_index, edge_attr, u,
                batch_e, batch_s, batch_t):
        """
        Sequentially applies: edge -> source -> target -> global updates.

        Returns:
            Tuple of updated (x_s, x_t, edge_attr, u).
        """
        if self.edge_model:
            edge_attr = self.edge_model(x_s, x_t, edge_index, edge_attr, u, batch_e)
        if self.s_model:
            x_s = self.s_model(x_s, x_t, edge_index, edge_attr, u, batch_s)
        if self.t_model:
            x_t = self.t_model(x_s, x_t, edge_index, edge_attr, u, batch_t)
        if self.global_model:
            u   = self.global_model(x_s, x_t, edge_index, edge_attr, u, batch_s, batch_t)
        return x_s, x_t, edge_attr, u


class GNN(torch.nn.Module):
    """
    Full GNN stacking multiple MetaLayer-style blocks to predict a discrete
    "time" value per edge via a differentiable rounding scheme.
    """
    def __init__(self):
        super(GNN, self).__init__()
        self.sharpness  = 20
        self.noiselevel = 0.3
        self.classes    = torch.arange(F_e_out).float().to(device)

        self.block_1     = Block(EdgeModel(), SModel(), TModel(), GlobalModel())
        self.bn_xs_1     = torch.nn.BatchNorm1d(F_xs)
        self.bn_xt_1     = torch.nn.BatchNorm1d(F_xt)
        self.bn_e_1      = torch.nn.BatchNorm1d(F_e)
        self.block_2     = Block(EdgeModel(), SModel(), TModel(), GlobalModel())
        self.bn_xs_2     = torch.nn.BatchNorm1d(F_xs)
        self.bn_xt_2     = torch.nn.BatchNorm1d(F_xt)
        self.bn_e_2      = torch.nn.BatchNorm1d(F_e)
        self.block_3     = Block(EdgeModel(), SModel(), TModel(), GlobalModel())
        self.bn_xs_3     = torch.nn.BatchNorm1d(F_xs)
        self.bn_xt_3     = torch.nn.BatchNorm1d(F_xt)
        self.bn_e_3      = torch.nn.BatchNorm1d(F_e)
        self.block_4     = Block(EdgeModel(), SModel(), TModel(), GlobalModel())
        self.bn_xs_4     = torch.nn.BatchNorm1d(F_xs)
        self.bn_xt_4     = torch.nn.BatchNorm1d(F_xt)
        self.bn_e_4      = torch.nn.BatchNorm1d(F_e)
        self.block_last  = Block(EdgeModel_out())

    def forward(self, data, batch_e, batch_s, batch_t, train=True):
        """
        Forward pass through four MetaLayer blocks plus final edge update.

        Args:
            data (BipartiteData): Batch of bipartite graphs.
            batch_e (LongTensor): Edge-to-graph mapping [E].
            batch_s (LongTensor): Source-node-to-graph mapping [N_s].
            batch_t (LongTensor): Target-node-to-graph mapping [N_t].
            train (bool): If True, apply noise+sigmoid rounding; else hard round.

        Returns:
            Tuple:
                time (Tensor): Predicted time per edge [E].
                edge_index (LongTensor): Unchanged edge_index for reference.
        """
        x_s, x_t = data.x_s, data.x_t
        edge_index, edge_attr, u = data.edge_index, data.edge_attr, data.u

        # Four rounds of MetaLayer-style updates with BatchNorm
        for blk, bn_xs, bn_xt, bn_e in [
            (self.block_1, self.bn_xs_1, self.bn_xt_1, self.bn_e_1),
            (self.block_2, self.bn_xs_2, self.bn_xt_2, self.bn_e_2),
            (self.block_3, self.bn_xs_3, self.bn_xt_3, self.bn_e_3),
            (self.block_4, self.bn_xs_4, self.bn_xt_4, self.bn_e_4),
        ]:
            x_s, x_t, edge_attr, u = blk(x_s, x_t, edge_index, edge_attr,
                                        u, batch_e, batch_s, batch_t)
            x_s = bn_xs(x_s); x_t = bn_xt(x_t); edge_attr = bn_e(edge_attr)

        # Final edge-only update
        x_s, x_t, edge_attr, u = self.block_last(
            x_s, x_t, edge_index, edge_attr, u, batch_e, batch_s, batch_t)

        prob = F.softmax(edge_attr, dim=-1)
        time = torch.sum(prob * self.classes, dim=-1)

        if train:
            noise = self.noiselevel * (torch.rand_like(time) - 0.5)
            time = time + noise
            intpart = torch.floor(time)
            time = intpart + torch.sigmoid(self.sharpness * (time - 0.5 - intpart))
        else:
            time = torch.round(time)

        return time, edge_index
