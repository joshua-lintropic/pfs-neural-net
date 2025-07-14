import torch
import torch_geometric
import torch.nn.functional as F
from torch_scatter import scatter, scatter_mean
from params import *

class BipartiteData(torch_geometric.data.Data):
    """
    Data class for a bipartite graph with separate source and target node sets.

    Attributes:
        edge_index (LongTensor): Edge indices [2, E], row0=src, row1=tgt.
        x_s (Tensor): Source-node features [S, F_s].
        x_t (Tensor): Target-node features [T, F_t].
        x_e (Tensor): Edge features [E, F_e].
        x_u (Tensor): Global features [B, F_u].
    """
    def __init__(self, edge_index, x_s, x_t, x_e, x_u):
        super(BipartiteData, self).__init__()
        if edge_index is not None:
            self.edge_index = edge_index.to(device)
        if x_s is not None:
            self.x_s = x_s.to(device)
        if x_t is not None:
            self.x_t = x_t.to(device)
            self.num_nodes = len(self.x_t)  # dummy to suppress warnings
        if x_e is not None:
            self.x_e = x_e.to(device)
        if x_u is not None:
            self.x_u = x_u.to(device)

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

class MLP(torch.nn.Sequential):
    def __init__(self, D1, D2, D3):
        super(MLP, self).__init__(
            torch.nn.Linear(D1, D2),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(D2, D3)
        )

class EdgeModel(MLP):
    """
    Edge update module: combines source/node features, target/node features,
    edge features, and global features into updated edge embeddings.
    """
    def __init__(self, Fdim=10, normed=True):
        F_message = 4 * Fdim # concatenate source, target, edge, global
        super(EdgeModel, self).__init__(F_message, F_message, Fdim)
        if normed:
            self.norm = torch.nn.BatchNorm1d(Fdim)
        else:
            self.norm = lambda x: x

    def forward(self, x_s, x_t, edge_index, edge_attr, u):
        """
        Args:
            x_s (Tensor): Source-node embeddings [S, Fdim].
            x_t (Tensor): Target-node embeddings [T, Fdim].
            edge_index (LongTensor): [2, E].
            edge_attr (Tensor): Edge embeddings [E, Fdim].
            u (Tensor): Global embeddings [B, Fdim].

        Returns:
            Tensor: Updated edge features [E, Fdim].
        """
        src, tgt = edge_index
        E = edge_attr.size(0)
        h = torch.cat([x_s[src], x_t[tgt], edge_attr, u.expand(E,-1)], dim=-1)
        return self.norm(super().forward(h))


class SModel(torch.nn.Module):
    """
    Source-node update: aggregates incoming edge messages (with statistics)
    and updates source-node embeddings.
    """
    def __init__(self, Fdim=10, normed=True):
        super(SModel, self).__init__()
        F_message = 2 * Fdim
        self.node_mlp_1 = MLP(F_message, F_message, F_message)

        F_message2 = 4 * F_message + 2 * Fdim
        self.node_mlp_2 = MLP(F_message2, F_message2, Fdim)

        if normed:
            self.norm = torch.nn.BatchNorm1d(Fdim)
        else:
            self.norm = lambda x: x


    def forward(self, x_s, x_t, edge_index, edge_attr, u):
        """
        Args:
            x_s (Tensor): Source embeddings [S, Fdim].
            x_t (Tensor): Target embeddings [T, Fdimd].
            edge_index (LongTensor): [2, E].
            edge_attr (Tensor): Edge embeddings [E, Fdim].
            u (Tensor): Global embeddings [B, Fdim].

        Returns:
            Tensor: Updated source-node features [S, Fdim].
        """
        src, tgt = edge_index
        msg = torch.cat([x_t[tgt], edge_attr], dim=1)
        msg = self.node_mlp_1(msg)

        # Count and aggregate stats of incoming messages
        mean = scatter(msg, src, dim=0, dim_size=x_s.size(0), reduce='mean')
        var = F.leaky_relu(scatter(msg**2, src, dim=0, dim_size=x_s.size(0), reduce='mean') - mean**2)
        std = torch.sqrt(var + 1e-6)
        skew = scatter((msg - mean[src])**3, src, dim=0, dim_size=x_s.size(0), reduce='mean') / std**3
        kurt = scatter((msg - mean[src])**4, src, dim=0, dim_size=x_s.size(0), reduce='mean') / std**4

        # zero out if nan
        mean = torch.nan_to_num(mean, nan=0.0)
        var  = torch.nan_to_num(var,  nan=0.0)
        std  = torch.sqrt(var + 1e-6)
        skew = torch.nan_to_num(skew, nan=0.0)
        kurt = torch.nan_to_num(kurt, nan=0.0)

        h_cat = torch.cat([x_s, mean, std, skew, kurt, u.expand(len(x_s), -1)], dim=-1)
        return self.norm(self.node_mlp_2(h_cat))


class TModel(torch.nn.Module):
    """
    Target-node update: sums incoming edge messages and updates target-node embeddings.
    """
    def __init__(self, Fdim=10, normed=True):
        super(TModel, self).__init__()
        F_message = 2 * Fdim
        self.node_mlp_1 = MLP(F_message, F_message, F_message)

        F_message2 = 4 * Fdim
        self.node_mlp_2 = MLP(F_message2, F_message2, Fdim)

        if normed:
            self.norm = torch.nn.BatchNorm1d(Fdim)
        else:
            self.norm = lambda x: x


    def forward(self, x_s, x_t, edge_index, edge_attr, u):
        """
        Args:
            x_s (Tensor): Source embeddings [S, Fdim].
            x_t (Tensor): Target embeddings [T, Fdim].
            edge_index (LongTensor): [2, E].
            edge_attr (Tensor): Edge embeddings [E, Fdim].
            u (Tensor): Global embeddings [B, Fdim].

        Returns:
            Tensor: Updated target-node features [T, Fdim].
        """
        src, tgt = edge_index
        msg = torch.cat([x_s[src], edge_attr], dim=1)
        msg = self.node_mlp_1(msg)
        agg = scatter(msg, tgt, dim=0, dim_size=x_t.size(0), reduce='sum')
        h_cat = torch.cat([x_t, agg, u.expand(len(x_t),-1)], dim=-1)
        return self.norm(self.node_mlp_2(h_cat))


class GlobalModel(MLP):
    """
    Graph-level update: pools node embeddings to update global features.
    """
    def __init__(self, Fdim=10, normed=True):
        F_message = 3 * Fdim
        super(GlobalModel, self).__init__(F_message, F_message, Fdim)
        if normed:
            self.norm = torch.nn.RMSNorm(Fdim)
        else:
            self.norm = lambda x: x


    def forward(self, x_s, x_t, edge_index, edge_attr, u):
        """
        Args:
            x_s (Tensor): Updated source-node features [S, Fdim].
            x_t (Tensor): Updated target-node features [T, Fdim].
            edge_index (LongTensor): [2, E].
            edge_attr (Tensor): Updated edge features [E, Fdim].
            u (Tensor): Previous global features [B, Fdim].

        Returns:
            Tensor: Updated global features [B, F_u].
        """
        s_mean = x_s.mean(dim=0, keepdim=True)
        t_mean = x_t.mean(dim=0, keepdim=True)
        h_cat = torch.cat([u, s_mean, t_mean], dim=-1)
        return self.norm(super().forward(h_cat))


class Block(torch.nn.Module):
    """
    A single MetaLayer block combining edge, source-node, target-node,
    and global update modules.
    """
    def __init__(self, Fdim=10, e_model=True, s_model=True, t_model=True, u_model=True, normed=True):
        super(Block, self).__init__()

        if e_model:
            self.edge_model = EdgeModel(Fdim, normed=normed)
        if s_model:
            self.s_model = SModel(Fdim, normed=normed)
        if t_model:
            self.t_model = TModel(Fdim, normed=normed)
        if u_model:
            self.global_model = GlobalModel(Fdim, normed=normed)

    def forward(self, args):
        """
        Sequentially applies: edge -> source -> target -> global updates.

        Returns:
            Tuple of updated (x_s, x_t, x_e, u).
        """
        edge_index, x_s, x_t, x_e, x_u = args
        if hasattr(self, "edge_model"):
            x_e = self.edge_model(x_s, x_t, edge_index, x_e, x_u)
        if hasattr(self, "s_model"):
            x_s = self.s_model(x_s, x_t, edge_index, x_e, x_u)
        if hasattr(self, "t_model"):
            x_t = self.t_model(x_s, x_t, edge_index, x_e, x_u)
        if hasattr(self, "global_model"):
            x_u = self.global_model(x_s, x_t, edge_index, x_e, x_u)
        return edge_index, x_s, x_t, x_e, x_u

class GNN(torch.nn.Module):
    """
    Full GNN stacking multiple MetaLayer-style blocks to predict a discrete
    "time" value per edge via a differentiable rounding scheme.
    """
    def __init__(self, B=4, Fdim=16, T=12, F_s=1, F_t=1, normed=True):
        super(GNN, self).__init__()

        # Encoders
        self.encoder_s = MLP(F_s, Fdim, Fdim)
        self.encoder_t = MLP(F_t, Fdim, Fdim)

        # Message passing blocks
        self.mpb = torch.nn.Sequential(*(Block(Fdim, normed=normed) for b in range(B)))

        # Decoders
        self.decoder_e = MLP(Fdim, Fdim, 1)
        self.decoder_s = MLP(Fdim, Fdim, T)

    def forward(self, graph):
        """
        Forward pass through four MetaLayer blocks plus final edge update.

        Args:
            graph (BipartiteData): Batch of bipartite graphs.

        Returns:
            Tuple:
                time (Tensor): Predicted time per edge [E].
                edge_index (LongTensor): Unchanged edge_index for reference.
        """
        # TODO: take a dimension argument `d`, lift x_s and x_t to this dimension
        x_s, x_t = graph.x_s, graph.x_t
        edge_index, x_e, x_u = graph.edge_index, graph.x_e, graph.x_u

        # Encode node features
        x_s = self.encoder_s(x_s)
        x_t = self.encoder_t(x_t)

        # B rounds of MetaLayer-style message passing on graph
        args = (edge_index, x_s, x_t, x_e, x_u)
        _, x_s, x_t, x_e, x_u = self.mpb(args)

        # make new graph
        return BipartiteData(edge_index, x_s, x_t, x_e, x_u)

    def edge_prediction(self, x_e, scale=1):
        # Decode predicted time/numbers from edges
        pred = self.decoder_e(x_e)
        pred = self.round(pred)
        pred = F.softplus(pred) * scale # ensure positive numbers, scaled for better initial guesses
        return pred

    def node_prediction(self, x_s, scale=1):
        # multi-class prediction for every fiber node
        pred = self.decoder_s(x_s) # look at this
        time = torch.softmax(pred, dim=-1) * scale # sum up to scale
        time = self.round(time) # TODO: does not need to sum up to scale!!!
        return time

    def round(self, x):
        if self.train:
            return x
        else:
            return torch.round(x)
