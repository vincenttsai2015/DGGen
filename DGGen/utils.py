"""
Collection of utility functions for training and evaluation scripts.
"""

import os
import torch
import itertools
import random
import numpy as np
import pandas as pd
from inspect import signature

# needed by load_model
from types import SimpleNamespace
from torch_geometric.nn import TGNMemory
from model import (
    GraphAttentionEmbedding,
    ProductLayer,
    MergeLayer,
    ReshapeLayer,
    col_RNN,
    LastNeighborLoader,
)
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
)
from torch_geometric.data import TemporalData

# DGB
import sys

sys.path.insert(0, "../")
from DGB.tgn.utils.utils import RandEdgeSampler, RandEdgeSampler_adversarial


def random_str(length=16):
    return ("%032x" % random.getrandbits(128))[:length]


class GetConfig:
    def __init__(self, config_path):
        self.config_path = config_path

    def __call__(self, attr):
        """
        Retrieves the queried attribute value from the config file. Loads the
        config file on first call.

        Parameters
        ----------
        attr : str
            Size of train+val+test sample
        fname : os.path, optional
            Path to config file, default is "./config.json"

        Returns
        -------
        Requested attribute
        """
        # if not hasattr(get_config, "config"):
        config = None
        with open(self.config_path) as f:
            config = eval(f.read())
        node = config
        for part in attr.split("."):
            node = node[part]
        return node


def preprocess(smp, device):
    # add start of sequence
    sos = torch.ones(smp.shape[:-1] + ((1,)), device=device)
    smp = torch.cat((sos, smp), -1)
    return smp.unsqueeze(-1)


def subsample_dst(
    pos_dst,
    rand_all_dst,
    n_sampled_dst,
    rand_smpdst,
    all_dst,
    min_dst_idx,
    tmap,
    device,
    sample_all=False,
):
    """
    Returns:
    `sampled_dst` size: (n_sampled_dst).
        contains all unique pos_dst (correct destinations) and other random (wrong) destinations
    `idx_pos_dst` size: (batch_size).
        element i contains the index of the positive destination of source i in sampled_dst
    """
    if sample_all or n_sampled_dst is None:
        return all_dst, pos_dst - min_dst_idx
    else:
        pos_dst_u = pos_dst.unique()
        rand_all_dst.random_()
        # make sure all pos_dst_u are included
        # (put negative values on the indexes of the pos_dst so that they are first when applying topk)
        rand_all_dst[pos_dst_u - min_dst_idx] = (
            -torch.arange(len(pos_dst_u), 0, -1).float().to(device)
        )
        # select a random sample
        _, rand_perm = rand_all_dst.topk(
            n_sampled_dst, dim=0, sorted=True, largest=False
        )
        # shuffle again
        rand_smpdst.random_()
        _, r = rand_smpdst.topk(n_sampled_dst, dim=0, sorted=True, largest=False)
        sampled_dst = all_dst[torch.scatter(rand_perm, dim=0, index=r, src=rand_perm)]
        # map pos_dst to the right indexes
        idx_pos_dst_u = r[
            : len(pos_dst_u)
        ]  # indexes of the correct destinations in `sampled_dst`
        tmap[pos_dst_u] = (
            idx_pos_dst_u  # pos_dst original index -> corresponding index in `sampled_dst`
        )
        idx_pos_dst = tmap[
            pos_dst
        ]  # translate the pos_dst original indexes to the new indexes in `sampled_dst`
        return sampled_dst, idx_pos_dst


def subsample_src(
    pos_src,
    all_src,
    rand_all_src,
    rand_smpsrc,
    tmap_src,
    n_sampled_src,
    device,
    sample_all=False,
):
    """
    Returns:
    `sampled_dst` size: (n_sampled_dst).
        contains all unique pos_dst (correct destinations) and other random (wrong) destinations
    `idx_pos_dst` size: (batch_size).
        element i contains the index of the positive destination of source i in sampled_dst
    """
    if sample_all or n_sampled_src is None:
        return all_src, pos_src
    else:
        pos_src_u = pos_src.unique()
        rand_all_src.random_()
        # make sure all pos_src_u are included
        # (put negative values on the indexes of the pos_src so that they are first when applying topk)
        rand_all_src[pos_src_u] = (
            -torch.arange(len(pos_src_u), 0, -1).float().to(device)
        )
        # select a random sample
        _, rand_perm = rand_all_src.topk(
            n_sampled_src, dim=0, sorted=True, largest=False
        )
        # shuffle again
        rand_smpsrc.random_()
        _, r = rand_smpsrc.topk(n_sampled_src, dim=0, sorted=True, largest=False)
        sampled_src = all_src[torch.scatter(rand_perm, dim=0, index=r, src=rand_perm)]
        # map pos_dst to the right indexes
        idx_pos_src_u = r[
            : len(pos_src_u)
        ]  # indexes of the correct destinations in `sampled_dst`
        tmap_src[pos_src_u] = (
            idx_pos_src_u  # pos_dst original index -> corresponding index in `sampled_dst`
        )
        idx_pos_src = tmap_src[
            pos_src
        ]  # translate the pos_dst original indexes to the new indexes in `sampled_dst`
        return sampled_src, idx_pos_src


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, epoch, checkpoint_dir):
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
    }

    filename = os.path.join(checkpoint_dir, "epoch={}.checkpoint.pth.tar".format(epoch))
    torch.save(state, filename)


def restore_checkpoint(model, checkpoint_dir, cuda=True, force=False, pretrain=False):
    """
    If a checkpoint exists, restores the PyTorch model from the checkpoint.
    Returns the model and the current epoch.
    """
    files = [
        fn
        for fn in os.listdir(checkpoint_dir)
        if fn.startswith("epoch=") and fn.endswith(".checkpoint.pth.tar")
    ]

    if not files:
        print("No saved models found")
        if force:
            raise Exception("Checkpoint not found")
        else:
            return model, 0

    # Find latest epoch
    for i in itertools.count(1):
        if "epoch={}.checkpoint.pth.tar".format(i) in files:
            epoch = i
        else:
            break

    if not force:
        print(
            f"Select epoch: Choose in range [0, {epoch}].",
            "Entering 0 will train from scratch.",
        )
        print(">> ", end="")
        in_epoch = int(input())
        if in_epoch not in range(epoch + 1):
            raise Exception("Invalid epoch number")
        if in_epoch == 0:
            print("Checkpoint not loaded")
            clear_checkpoint(checkpoint_dir)
            return model, 0
    else:
        print(f"Select epoch: Choose in range [1, {epoch}].")
        in_epoch = int(input())
        if in_epoch not in range(1, epoch + 1):
            raise Exception("Invalid epoch number")

    filename = os.path.join(checkpoint_dir, f"epoch={in_epoch}.checkpoint.pth.tar")

    print("Loading from checkpoint {}?".format(filename))

    if cuda:
        checkpoint = torch.load(filename)
    else:
        checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)

    try:
        if pretrain:
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint["state_dict"])
        print(
            "=> Successfully restored checkpoint (trained for {} epochs)".format(
                checkpoint["epoch"]
            )
        )
    except:
        print("=> Checkpoint not successfully restored")
        raise

    return model, in_epoch


def clear_checkpoint(checkpoint_dir):
    fnames = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth.tar")]
    for f in fnames:
        os.remove(os.path.join(checkpoint_dir, f))

    print("Checkpoint removed")


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    return


def train_val1_val2_test_split(
    data, val_1_ratio: float = 0.10, val_2_ratio: float = 0.10, test_ratio: float = 0.10
):
    r"""Splits the data in training, validation and test sets according to
    time.

    Args:
        val_ratio (float, optional): The proportion (in percents) of the
            dataset to include in the validation split.
            (default: :obj:`0.15`)
        test_ratio (float, optional): The proportion (in percents) of the
            dataset to include in the test split. (default: :obj:`0.15`)
    """
    val_1_time, val_2_time, test_time = np.quantile(
        data.t.cpu().numpy(),
        [
            1.0 - val_1_ratio - val_2_ratio - test_ratio,
            1.0 - val_2_ratio - test_ratio,
            1.0 - test_ratio,
        ],
    )

    val_1_idx = int((data.t <= val_1_time).sum())
    val_2_idx = int((data.t <= val_2_time).sum())
    test_idx = int((data.t <= test_time).sum())

    return (
        data[:val_1_idx],
        data[val_1_idx:val_2_idx],
        data[val_2_idx:test_idx],
        data[test_idx:],
    )


def save_model(best_model, path, config_dict, neighbor_loader):
    os.makedirs(path[: path.rindex(os.path.sep)], exist_ok=True)
    torch.save(
        {
            "memory_state_dict": best_model[0].state_dict(),
            "gnn_state_dict": best_model[1].state_dict(),
            "embd_to_score_dst_state_dict": best_model[2].state_dict(),
            "embd_to_score_src_state_dict": best_model[3].state_dict(),
            "feats_model_state_dict": best_model[4].state_dict(),
            "embd_to_h0_state_dict": best_model[5].state_dict(),
            "config_dict": config_dict,
            "neighbor_loader": neighbor_loader,
        },
        path,
    )


def load_state_dict_and_update_keys(module, state_dict):
    missing_keys = module.state_dict().keys() - state_dict.keys()
    unexpected_keys = state_dict.keys() - module.state_dict().keys()
    if len(missing_keys) == len(unexpected_keys) == 1:
        print(
            f"Replacing missing_keys {missing_keys} with unexpected_keys {unexpected_keys} in {module} state_dict"
        )
        state_dict[missing_keys.pop()] = state_dict.pop(unexpected_keys.pop())
    module.load_state_dict(state_dict)


def load_model(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(path, "rb") as f:
        checkpoint = torch.load(f, map_location=torch.device(device))
    config_dict = checkpoint["config_dict"]
    cfg = SimpleNamespace(**config_dict)

    data = config_dict["data"].to(device)

    memory = TGNMemory(
        data.num_nodes,
        data.msg.size(-1),
        cfg.memory_dim,
        cfg.time_dim,
        message_module=IdentityMessage(data.msg.size(-1), cfg.memory_dim, cfg.time_dim),
        aggregator_module=LastAggregator(),
    ).to(device)

    gnn = GraphAttentionEmbedding(
        in_channels=cfg.memory_dim,
        out_channels=cfg.embedding_dim,
        msg_dim=data.msg.size(-1),
        time_enc=memory.time_enc,
    ).to(device)

    embd_to_score_dst = ProductLayer(in_channels=cfg.embedding_dim).to(device)

    embd_to_score_src = ReshapeLayer(in_channels=cfg.embedding_dim, out_channels=1).to(
        device
    )

    feats_model = col_RNN(
        len(cfg.col2K),
        cfg.col2K,
        embed_dim=1,
        hidden_size=cfg.feats_model_h_dim,
        num_layers=1,
        n_comp=cfg.n_comp,
    ).to(device)

    embd_to_h0 = MergeLayer(
        in_channels=cfg.embedding_dim, out_channels=cfg.feats_model_h_dim
    ).to(device)

    load_state_dict_and_update_keys(memory, checkpoint["memory_state_dict"])
    load_state_dict_and_update_keys(gnn, checkpoint["gnn_state_dict"])
    load_state_dict_and_update_keys(
        embd_to_score_dst, checkpoint["embd_to_score_dst_state_dict"]
    )
    load_state_dict_and_update_keys(
        embd_to_score_src, checkpoint["embd_to_score_src_state_dict"]
    )
    load_state_dict_and_update_keys(feats_model, checkpoint["feats_model_state_dict"])
    load_state_dict_and_update_keys(embd_to_h0, checkpoint["embd_to_h0_state_dict"])
    neighbor_loader = checkpoint["neighbor_loader"]

    memory.to(device)
    gnn.to(device)
    embd_to_score_dst.to(device)
    embd_to_score_src.to(device)
    feats_model.to(device)
    embd_to_h0.to(device)

    for n, v in memory.msg_s_store.items():
        memory.msg_s_store[n] = tuple(
            memory.msg_s_store[n][i].to(device) for i in range(len(v))
        )
    for n, v in memory.msg_d_store.items():
        memory.msg_d_store[n] = tuple(
            memory.msg_d_store[n][i].to(device) for i in range(len(v))
        )

    return (
        config_dict,
        memory,
        gnn,
        embd_to_score_dst,
        embd_to_score_src,
        feats_model,
        embd_to_h0,
        neighbor_loader,
    )


def save_tgn():
    pass


def load_tgn():
    pass


def temporaldata_to_df(
    gen_data, columns=None, dtypes=None, st=0, en=None, w_attr=False
):
    if columns is None:
        columns = ["src", "dst", "t"] + [
            f"msg_{i}" for i in range(gen_data.msg.shape[1])
        ]
    if dtypes is None:
        dtypes = [int, int, int] + [float for _ in range(gen_data.msg.shape[1])]
        dtypes = dict(zip(columns, dtypes))

    gen_data = gen_data[st:en]
    if w_attr:
        attr_cols = [
            gen_data.global_attr.cpu().numpy(),
            gen_data.src_attr.cpu().numpy(),
            gen_data.dst_attr.cpu().numpy(),
        ]
        attr_cols_flat = list(
            np.concatenate([list(c.T) if len(c.shape) > 1 else [c] for c in attr_cols])
        )
    else:
        attr_cols_flat = []

    df0 = pd.DataFrame(
        np.stack(
            [
                gen_data.src.cpu().numpy(),
                gen_data.dst.cpu().numpy(),
                gen_data.t.cpu().numpy(),
            ]
            + list(gen_data.msg[:, :].cpu().numpy().T)
            + attr_cols_flat,
            1,
        ),
        columns=columns,
    ).astype(dtypes)
    return df0


def convert_dgb_data_to_pyg_TemporalData(
    dgb_data, min_dgb_edge_idx=1, pyg_temporaldata=None
):
    dgb_data_e_idxs = torch.tensor(dgb_data.edge_idxs - min_dgb_edge_idx)
    if pyg_temporaldata is not None:
        dgb_data_e_idxs = dgb_data_e_idxs.to(pyg_temporaldata.src.device.type)
        return pyg_temporaldata[dgb_data_e_idxs]
    else:
        print("Not implemented")


def assess_new_node_data(val_data, new_node_val_data):
    min_val_idx = val_data.edge_id.min()
    assert (
        val_data.edge_id - min_val_idx
        == torch.arange(
            val_data.num_events, dtype=torch.long, device=val_data.edge_id.device.type
        )
    ).all(), "data.edge_id has gaps."
    assert (
        val_data.edge_id.min()
        <= new_node_val_data.edge_id.min()
        <= new_node_val_data.edge_id.max()
        <= val_data.edge_id.max()
    ), "new_node_data is not contained in data"
    assert (val_data.edge_id == val_data.edge_id.sort()[0]).all(), "data is not sorted"
    assert (
        new_node_val_data.edge_id == new_node_val_data.edge_id.sort()[0]
    ).all(), "new_node_data is not sorted"


def get_full_ngh_loader(full_data, train_data, num_neighbors, device):
    neighbor_loader = LastNeighborLoader(
        full_data.num_nodes, size=num_neighbors, device=device
    )
    # get the edge_idx of the last interaction of the training dataset
    last_train_data_edge_id = train_data.edge_id.max()
    # get all edge_idxs before the last interaction of the training dataset
    e_id_full_data = full_data.edge_id[full_data.edge_id < last_train_data_edge_id]
    # add those interactions
    neighbor_loader.insert(full_data[e_id_full_data].src, full_data[e_id_full_data].dst)
    return neighbor_loader


def get_indicator_tensor_of_new_nodes(data, new_node_data):
    min_idx = data.edge_id.min()
    has_new_node = torch.zeros_like(data.edge_id)
    has_new_node[new_node_data.edge_id - min_idx] = 1.0
    return has_new_node


class FromDGBtoPYG:
    def __init__(self, dgb_data, pyg_data):
        self.device = pyg_data.src.device.type
        self.dgb_data = dgb_data
        self.pyg_data = pyg_data
        self.src_offset = dgb_data.sources.min() - pyg_data.src.min().item()
        self.dst_offset = dgb_data.destinations.min() - pyg_data.dst.min().item()
        self.test_node_id_conversion()

    def test_node_id_conversion(self):
        assert (
            torch.tensor(self.dgb_data.sources, device=self.device) - self.src_offset
            == self.pyg_data.src
        ).all() or "src node IDs cannot be converted"
        assert (
            torch.tensor(self.dgb_data.destinations, device=self.device)
            - self.dst_offset
            == self.pyg_data.dst
        ).all() or "dst node IDs cannot be converted"

    def to_pyg_src(self, sources):
        return (
            torch.tensor(sources, dtype=torch.long, device=self.device)
            - self.src_offset
        )

    def to_pyg_dst(self, destinations):
        return (
            torch.tensor(destinations, dtype=torch.long, device=self.device)
            - self.dst_offset
        )

    def convert_src_and_dst(self, sources, destinations):
        return self.to_pyg_src(sources), self.to_pyg_dst(destinations)

    def sampler_converter_decorator(self):
        def wrapper_sampler(sampler):
            def wrapper(*args):
                num_args = len(signature(sampler).parameters)
                out = sampler(*args[:num_args])
                # print(out)
                return self.convert_src_and_dst(*out)

            return wrapper

        return wrapper_sampler


def initialize_test_negative_edge_sampler(
    NEG_SAMPLE, full_data, val_data, new_node_test_data
):
    if NEG_SAMPLE != "rnd":
        test_rand_sampler = RandEdgeSampler_adversarial(
            full_data.sources,
            full_data.destinations,
            full_data.timestamps,
            val_data.timestamps[-1],
            NEG_SAMPLE,
            seed=2,
        )
        nn_test_rand_sampler = RandEdgeSampler_adversarial(
            new_node_test_data.sources,
            new_node_test_data.destinations,
            new_node_test_data.timestamps,
            val_data.timestamps[-1],
            NEG_SAMPLE,
            seed=3,
        )
    else:
        test_rand_sampler = RandEdgeSampler(
            full_data.sources, full_data.destinations, seed=2
        )
        nn_test_rand_sampler = RandEdgeSampler(
            new_node_test_data.sources, new_node_test_data.destinations, seed=3
        )
    return test_rand_sampler, nn_test_rand_sampler
