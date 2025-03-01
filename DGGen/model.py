import torch
from torch_geometric.nn import TransformerConv
from torch.nn import Linear
from typing import Tuple
from torch import Tensor


class LastNeighborLoader:
    def __init__(self, num_nodes: int, size: int, device=None):
        self.size = size

        self.neighbors = torch.empty((num_nodes, size), dtype=torch.long, device=device)
        self.e_id = torch.empty((num_nodes, size), dtype=torch.long, device=device)
        self._assoc = torch.empty(num_nodes, dtype=torch.long, device=device)

        self.reset_state()

    def __call__(self, n_id: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        neighbors = self.neighbors[n_id]
        nodes = n_id.view(-1, 1).repeat(1, self.size)
        e_id = self.e_id[n_id]

        # Filter invalid neighbors (identified by `e_id < 0`).
        mask = e_id >= 0
        neighbors, nodes, e_id = neighbors[mask], nodes[mask], e_id[mask]

        # Relabel node indices.
        n_id = torch.cat([n_id, neighbors]).unique()
        self._assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)
        neighbors, nodes = self._assoc[neighbors], self._assoc[nodes]

        return n_id, torch.stack([neighbors, nodes]), e_id

    def insert(self, src: Tensor, dst: Tensor):
        # Inserts newly encountered interactions into an ever-growing
        # (undirected) temporal graph.

        # Collect central nodes, their neighbors and the current event ids.
        neighbors = torch.cat([src, dst], dim=0)
        nodes = torch.cat([dst, src], dim=0)
        e_id = torch.arange(
            self.cur_e_id, self.cur_e_id + src.size(0), device=src.device
        ).repeat(2)
        self.cur_e_id += src.numel()

        # Convert newly encountered interaction ids so that they point to
        # locations of a "dense" format of shape [num_nodes, size].
        nodes, perm = nodes.sort(stable=True)
        neighbors, e_id = neighbors[perm], e_id[perm]

        n_id = nodes.unique()
        self._assoc[n_id] = torch.arange(n_id.numel(), device=n_id.device)

        dense_id = torch.arange(nodes.size(0), device=nodes.device) % self.size
        dense_id += self._assoc[nodes].mul_(self.size)

        dense_e_id = e_id.new_full((n_id.numel() * self.size,), -1)
        dense_e_id[dense_id] = e_id
        dense_e_id = dense_e_id.view(-1, self.size)

        dense_neighbors = e_id.new_empty(n_id.numel() * self.size)
        dense_neighbors[dense_id] = neighbors
        dense_neighbors = dense_neighbors.view(-1, self.size)

        # Collect new and old interactions...
        e_id = torch.cat([self.e_id[n_id, : self.size], dense_e_id], dim=-1)
        neighbors = torch.cat(
            [self.neighbors[n_id, : self.size], dense_neighbors], dim=-1
        )

        # And sort them based on `e_id`.
        e_id, perm = e_id.topk(self.size, dim=-1)
        self.e_id[n_id] = e_id
        self.neighbors[n_id] = torch.gather(neighbors, 1, perm)

    def reset_state(self):
        self.cur_e_id = 0
        self.e_id.fill_(-1)


class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(
            in_channels, out_channels // 2, heads=2, dropout=0.1, edge_dim=edge_dim
        )

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h)


class ProductLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, out_channels)

    def forward(self, z_src, z_dst):
        """Return: tensor of shape (z_src.shape[0], z_dst.shape[0])
        where element (i, j) is the unnormalized score
        for an interaction between src `i` and dst `j`
        """
        b = self.lin_src(z_src)
        a = self.lin_dst(z_dst)
        h = a + b.unsqueeze(1)
        h = h.relu()
        return self.lin_final(h).squeeze(2)


class MergeLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, out_channels)

    def forward(self, z_src, z_dst):
        """Return: tensor of shape z_src.shape = z_dst.shape = (batch_size, out_channels)
        where element `i` is the transformed embedding obtained combining z_src[i] and z_dst[i]
        """
        b = self.lin_src(z_src).relu()
        a = self.lin_dst(z_dst).relu()
        h = a + b
        h = h.relu()
        return self.lin_final(h)


class ReshapeLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, out_channels)

    def forward(self, z):
        """
        Takes a tensor of shape (z.shape[0], in_channels)
        Return: tensor of shape (z.shape[0], out_channels)
        """
        h = self.lin_src(z).relu()
        return self.lin_final(h)


class Column:
    def __init__(self, idx, seq_len, distrib, params):
        """Class to describe each element in the sequence (column)

        idx: int, index of the element in the sequence (0 is the first)

        seq_len: int, total length of sequence (number of columns)

        distrib: class of `torch.distributions`

        params: dict of distribution parameters and how to compute them.
            key is a string with the name of a parameter of `distrib`
            value is a callable with input the last layer of the `model.decoder` that outputs the value of the parameter

        Example:

            Column(0,
                    distrib=torch.distributions.Normal,
                    params={"loc": model.layer_mu,
                            "scale": lambda z: model.layer_std(z).abs()})
        """
        super(Column, self).__init__()
        self.idx = idx
        self.seq_len = seq_len
        self.mask = self.col2mask()
        self.distrib = distrib
        self.params = params

    def col2mask(self):
        mask = torch.zeros(self.seq_len).bool()
        mask[self.idx] = True
        return mask

    def ptdist(self, x):
        #         return self.distrib(**{p: l(x[:, self.mask]) for p, l in self.params.items()})
        params = {
            p: (
                l[0](**{p1: l1(x[:, self.mask]) for p1, l1 in l[1].items()})
                if isinstance(l, tuple)
                else l(x[:, self.mask])
            )
            for p, l in self.params.items()
        }
        return self.distrib(**params)


class MLP(torch.nn.Module):
    def __init__(self, in_channels, out_channels, h_channels=None):
        super(MLP, self).__init__()
        if h_channels is None:
            h_channels = in_channels
        self.lin_first = torch.nn.Linear(in_channels, h_channels)
        self.lin_final = torch.nn.Linear(h_channels, out_channels)

    def forward(self, x):
        x = self.lin_first(x).relu()
        return self.lin_final(x)


class Add(torch.nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return x + self.epsilon


class col_RNN(torch.nn.Module):
    # define model elements
    def __init__(
        self, seq_len, col2K, embed_dim=1, hidden_size=8, num_layers=1, n_comp=3
    ):
        super(col_RNN, self).__init__()

        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.decoder = torch.nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )

        # exponential
        self.layer_exp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 1), torch.nn.Softplus(), Add()
        )
        # normal
        self.layer_mu = torch.nn.Linear(hidden_size, 1)
        self.layer_std = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 1), torch.nn.Softplus(), Add()
        )
        # gmm
        self.layer_gmm_mix = torch.nn.Linear(hidden_size, n_comp)
        self.layer_gmm_mu = torch.nn.Linear(hidden_size, n_comp)
        self.layer_gmm_std = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, n_comp), torch.nn.Softplus(), Add()
        )

        # create columns
        self.categorical_col_modules = torch.nn.ModuleList([])
        self.col2K = col2K
        self.columns = [self.create_column(self.col2K[i], i) for i in range(seq_len)]
        self.categorical_idx = [i for i, v in col2K.items() if isinstance(v, int)]

    def create_column(self, col, i):
        if isinstance(col, int):
            self.categorical_col_modules.append(
                torch.nn.Linear(self.hidden_size, self.col2K[i])
            )
            column = Column(
                i,
                self.seq_len,
                distrib=torch.distributions.Categorical,
                params={"logits": self.categorical_col_modules[-1]},
            )

        elif col == "exponential":
            column = Column(
                i,
                self.seq_len,
                distrib=torch.distributions.Exponential,
                params={"rate": self.layer_exp},
            )

        elif col == "normal":
            column = Column(
                i,
                self.seq_len,
                distrib=torch.distributions.Normal,
                params={"loc": self.layer_mu, "scale": self.layer_std},
            )
        else:
            column = Column(
                i,
                self.seq_len,
                distrib=torch.distributions.MixtureSameFamily,
                params={
                    "mixture_distribution": (
                        torch.distributions.Categorical,
                        {"logits": self.layer_gmm_mix},
                    ),
                    "component_distribution": (
                        torch.distributions.Normal,
                        {"loc": self.layer_gmm_mu, "scale": self.layer_gmm_std},
                    ),
                },
            )
        return column

    # forward propagate input
    def forward(self, x, hx=None):
        x, _ = self.decoder(x, hx)
        return x

    def sample(self, num_samples, preprocess, hx=None, device="cuda"):
        seq_len = self.seq_len

        if hx is not None:
            assert (
                hx.shape[1] == num_samples
            ), "Dimension 1 of `hx` should match `num_samples`."

        with torch.no_grad():
            y = torch.empty((num_samples, seq_len)).float().to(device)
            y = preprocess(y, device=device)

            for i, c in enumerate(self.columns):
                x = self(y[:, :-1], hx)
                a = c.ptdist(x).sample()
                if isinstance(c.ptdist(x), torch.distributions.Normal) or isinstance(
                    c.ptdist(x), torch.distributions.Exponential
                ):
                    y[:, i + 1, :] = a[:, 0, :]
                else:
                    y[:, i + 1, :] = a
        y0 = y[:, 1:, 0]
        return y0


def preprocess(smp, device):
    # add start of sequence
    sos = torch.ones(smp.shape[:-1] + ((1,))).to(device)
    smp = torch.cat((sos, smp), -1).to(device)
    return smp.unsqueeze(-1)
