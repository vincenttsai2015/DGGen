import numpy as np
import pandas as pd


import torch

from torch_geometric.datasets import JODIEDataset
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
)

from bikeshare import get_cbs_data
import utils
from utils import GetConfig, setup_seed, save_model, load_model
from model import (
    GraphAttentionEmbedding,
    ProductLayer,
    MergeLayer,
    ReshapeLayer,
    col_RNN,
    LastNeighborLoader,
)
from train_epoch import train_epoch

from val_epoch import validate
from utils import (
    convert_dgb_data_to_pyg_TemporalData,
    assess_new_node_data,
    get_full_ngh_loader,
    get_indicator_tensor_of_new_nodes,
)

# DGB
import sys

sys.path.insert(0, "../")
from DGB.tgn.utils.data_processing import get_data  # type: ignore


def main(config_path=None):

    assert config_path is not None, "Valid config path required as input to main."

    run_id = utils.random_str(length=4)
    now = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S") + f"_{run_id}"

    get_config = GetConfig(config_path=config_path)

    hyperparams = dict(
        name=get_config("name"),
        n_gpu=get_config("n_gpu"),
        num_neighbors=get_config("num_neighbors"),
        batch_size=get_config("batch_size"),
        n_sampled_src=get_config("n_sampled_src"),
        n_sampled_dst=get_config("n_sampled_dst"),
        memory_dim=get_config("memory_dim"),
        time_dim=get_config("time_dim"),
        embedding_dim=get_config("embedding_dim"),
        feats_model_h_dim=get_config("feats_model_h_dim"),
        data_name=get_config("data_name"),
        data_path=get_config("data_path"),
        lr=get_config("lr"),
        threshold_eps=get_config("threshold_eps"),
        threshold_epochs=get_config("threshold_epochs"),
        eps=get_config("eps"),
        num_feats=get_config("num_feats"),
        seed=get_config("seed"),
        n_comp=get_config("n_comp"),
        num_epochs=get_config("num_epochs"),
    )

    # setup seed
    setup_seed(hyperparams["seed"])

    # args
    num_neighbors = hyperparams["num_neighbors"]
    batch_size = hyperparams["batch_size"]

    n_sampled_src = hyperparams["n_sampled_src"]
    n_sampled_dst = hyperparams["n_sampled_dst"]
    memory_dim = hyperparams["memory_dim"]
    time_dim = hyperparams["time_dim"]
    embedding_dim = hyperparams["embedding_dim"]

    feats_model_h_dim = hyperparams["feats_model_h_dim"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    name = hyperparams["name"]
    path = hyperparams["data_path"]
    print(f"Training {name} on {hyperparams['data_name']} dataset at path {path}. ")

    data = None
    if hyperparams["data_name"] in [
        "wikipedia",
        "wiki_small",
        "reddit",
        "mooc",
        "lastfm",
    ]:
        data_name = (
            "wikipedia"
            if hyperparams["data_name"] == "wiki_small"
            else hyperparams["data_name"]
        )
        dataset = JODIEDataset(path, name=data_name)
        data = dataset[0]
        if hyperparams["data_name"] == "wiki_small":
            print("Truncating wiki -> wiki_small")
            data = data[:2981]
        data.edge_id = torch.arange(data.num_edges, dtype=torch.long)
        data = data.to(device)
        # reduce the number of features
        data.msg = data.msg[:, :12]
        reduce_feats = False
        if reduce_feats is True:
            sampled_feat_idxs = torch.randint(
                0, data.msg.shape[1], (hyperparams["num_feats"],)
            )
            data.msg = data.msg[:, sampled_feat_idxs]
        col2K = {i: "gmm" for i in range(data.msg.shape[1])}
    elif hyperparams["data_name"] == "cbs":
        data, col2K = get_cbs_data(path=path)
        data.edge_id = torch.arange(data.num_edges, dtype=torch.long)
        data = data.to(device)
    else:
        exit(f"Unknown data_name {hyperparams['data_name']}. Terminating.")

    # add time diff
    dt = torch.cat([torch.tensor([0.0], device=device), data.t[1:] - data.t[:-1]])
    data.msg = torch.cat([dt.unsqueeze(1), data.msg], dim=1)
    # data.msg[:2]
    col2K = {**{0: "exponential"}, **{k + 1: v for k, v in col2K.items()}}
    print(col2K)

    # Ensure to only sample actual destination nodes as negatives.
    min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())
    min_src_idx, max_src_idx = int(data.src.min()), int(data.src.max())

    n_sampled_src = (
        n_sampled_src if n_sampled_src is not None else max_src_idx - min_src_idx
    )
    n_sampled_dst = (
        n_sampled_dst if n_sampled_dst is not None else max_dst_idx - min_dst_idx
    )

    # DGB
    # Extract data for training, validation and testing
    val_ratio, test_ratio = 0.15, 0.15
    different_new_nodes = False
    randomize_features = False
    (
        node_features,
        edge_features,
        full_data,
        train_data,
        val_data,
        test_data,
        new_node_val_data,
        new_node_test_data,
    ) = get_data(
        hyperparams["data_name"],
        val_ratio,
        test_ratio,
        different_new_nodes_between_val_and_test=different_new_nodes,
        randomize_features=randomize_features,
    )

    train_temp_data = convert_dgb_data_to_pyg_TemporalData(
        train_data, min_dgb_edge_idx=full_data.edge_idxs.min(), pyg_temporaldata=data
    )

    val_temp_data = convert_dgb_data_to_pyg_TemporalData(
        val_data, min_dgb_edge_idx=full_data.edge_idxs.min(), pyg_temporaldata=data
    )

    new_node_val_temp_data = convert_dgb_data_to_pyg_TemporalData(
        new_node_val_data,
        min_dgb_edge_idx=full_data.edge_idxs.min(),
        pyg_temporaldata=data,
    )

    test_temp_data = convert_dgb_data_to_pyg_TemporalData(
        test_data, min_dgb_edge_idx=full_data.edge_idxs.min(), pyg_temporaldata=data
    )

    new_node_test_temp_data = convert_dgb_data_to_pyg_TemporalData(
        new_node_test_data,
        min_dgb_edge_idx=full_data.edge_idxs.min(),
        pyg_temporaldata=data,
    )

    assess_new_node_data(val_temp_data, new_node_val_temp_data)
    assess_new_node_data(test_temp_data, new_node_test_temp_data)

    # add indicator of new nodes to val and test datasets
    has_new_node = get_indicator_tensor_of_new_nodes(
        val_temp_data, new_node_val_temp_data
    )
    val_temp_data.new_node = has_new_node
    has_new_node = get_indicator_tensor_of_new_nodes(
        test_temp_data, new_node_test_temp_data
    )
    test_temp_data.new_node = has_new_node

    train_loader = TemporalDataLoader(train_temp_data, batch_size=batch_size)
    val_loader = TemporalDataLoader(val_temp_data, batch_size=batch_size)
    test_loader = TemporalDataLoader(test_temp_data, batch_size=batch_size)

    neighbor_loader = LastNeighborLoader(
        data.num_nodes,
        size=num_neighbors,
        device=device,
    )

    memory = TGNMemory(
        data.num_nodes,
        data.msg.size(-1),
        memory_dim,
        time_dim,
        message_module=IdentityMessage(data.msg.size(-1), memory_dim, time_dim),
        aggregator_module=LastAggregator(),
    ).to(device)

    gnn = GraphAttentionEmbedding(
        in_channels=memory_dim,
        out_channels=embedding_dim,
        msg_dim=data.msg.size(-1),
        time_enc=memory.time_enc,
    ).to(device)

    embd_to_score_dst = ProductLayer(in_channels=embedding_dim).to(device)

    embd_to_score_src = ReshapeLayer(in_channels=embedding_dim, out_channels=1).to(
        device
    )

    feats_model = col_RNN(
        data.msg.shape[1],
        col2K,
        embed_dim=1,
        hidden_size=feats_model_h_dim,
        num_layers=1,
        n_comp=hyperparams["n_comp"],
    ).to(device)

    embd_to_h0 = MergeLayer(
        in_channels=embedding_dim, out_channels=feats_model_h_dim
    ).to(device)

    # OPTIM
    optimizer = torch.optim.Adam(
        set(memory.parameters())
        | set(gnn.parameters())
        | set(embd_to_score_src.parameters())
        | set(embd_to_score_dst.parameters())
        | set(feats_model.parameters())
        | set(embd_to_h0.parameters()),
        lr=hyperparams["lr"],
    )

    # Helper vector to map global node indices to local ones.
    assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)  # type: ignore

    all_dst = torch.arange(min_dst_idx, max_dst_idx + 1).to(device)

    len_all_dst = len(all_dst)

    rand_all_dst = torch.rand(len_all_dst).to(device)
    rand_smpdst = (
        torch.rand(n_sampled_dst).to(device) if n_sampled_dst is not None else None
    )
    tmap = torch.empty(max_dst_idx + 1, dtype=torch.long).to(device)

    all_src = torch.arange(min_src_idx, max_src_idx + 1).to(device)

    len_all_src = len(all_src)

    rand_all_src = torch.rand(len_all_src).to(device)
    rand_smpsrc = (
        torch.rand(n_sampled_src).to(device) if n_sampled_src is not None else None
    )
    tmap_src = torch.empty(max_src_idx + 1, dtype=torch.long).to(device)

    """
    BEGIN TRAIN LOOP
    """
    results = []
    results_feats = []

    best_loss = 1e8
    best_model = None

    tot_epochs = 0

    eps = hyperparams["eps"]
    threshold_eps = hyperparams["threshold_eps"]
    min_eps = eps * 0.01
    threshold_epochs = hyperparams["threshold_epochs"]
    num_epochs = hyperparams["num_epochs"]

    best_val_loss = 1000
    best_train_loss = 1000
    best_ds_val_aucs = 0
    best_ds_val_aps = 0

    for epoch in range(1, num_epochs + 1):
        tot_epochs += 1

        mean_loss, mean_loss_feats, best_model, best_loss = train_epoch(
            memory,
            gnn,
            embd_to_score_dst,
            feats_model,
            embd_to_h0,
            neighbor_loader,
            train_loader,
            device,
            all_src,
            rand_all_src,
            rand_smpsrc,
            tmap_src,
            n_sampled_src,
            rand_all_dst,
            n_sampled_dst,
            rand_smpdst,
            all_dst,
            min_dst_idx,
            tmap,
            optimizer,
            best_loss,
            assoc,
            data,
            eps,
            embd_to_score_src,
            train_temp_data,
            max_dst_idx,
        )

        if mean_loss < best_train_loss:
            best_train_loss = mean_loss

            best_model = []
            best_model.append(memory)
            best_model.append(gnn)
            best_model.append(embd_to_score_dst)
            best_model.append(embd_to_score_src)
            best_model.append(feats_model)
            best_model.append(embd_to_h0)

            dataset_ = get_config("data_name")
            config_dict = {
                "num_neighbors": hyperparams["num_neighbors"],
                "batch_size": hyperparams["batch_size"],
                "memory_dim": hyperparams["memory_dim"],
                "time_dim": hyperparams["time_dim"],
                "embedding_dim": hyperparams["embedding_dim"],
                "feats_model_h_dim": hyperparams["feats_model_h_dim"],
                "n_comp": hyperparams["n_comp"],
                "lr": hyperparams["lr"],
                "eps": hyperparams["eps"],
                "threshold_eps": hyperparams["threshold_eps"],
                "min_eps": min_eps,
                "threshold_epochs": hyperparams["threshold_epochs"],
                "seed": hyperparams["seed"],
                "data": data,
                "min_dst_idx": min_dst_idx,
                "max_dst_idx": max_dst_idx,
                "all_src": all_src,
                "all_dst": all_dst,
                "rand_all_src": rand_all_src,
                "rand_smpsrc": rand_smpsrc,
                "tmap_src": tmap_src,
                "n_sampled_src": n_sampled_src,
                "rand_all_dst": rand_all_dst,
                "n_sampled_dst": n_sampled_dst,
                "rand_smpdst": rand_smpdst,
                "tmap": tmap,
                "assoc": assoc,
                "val_data": test_data,
                "loader": test_loader,
                "col2K": col2K,
                "best_train_loss_epoch": epoch,
            }
            save_model(
                best_model,
                f"./saved_models/train_loss/{dataset_}_{name}_{now}.pt",
                config_dict,
                neighbor_loader,
            )

        # Validation
        with torch.no_grad():
            # Validation uses the full graph
            # create a LastNeighborLoader with all interactions until the last training interaction
            neighbor_loader = get_full_ngh_loader(
                data, train_temp_data, num_neighbors, device
            )
            (
                mean_val_loss,
                mean_val_loss_feats,
                best_val_model_,
                best_val_loss_,
                metrics_dict,
            ) = validate(
                embd_to_score_src,
                embd_to_score_dst,
                memory,
                gnn,
                feats_model,
                embd_to_h0,
                neighbor_loader,
                min_dst_idx,
                max_dst_idx,
                all_src,
                all_dst,
                rand_all_src,
                rand_smpsrc,
                tmap_src,
                n_sampled_src,
                rand_all_dst,
                n_sampled_dst,
                rand_smpdst,
                tmap,
                assoc,
                data,
                eps,
                loader=val_loader,
                compute_loss=True,
                test_rand_samplers=None,
                nn_test_rand_samplers=None,
            )

        if metrics_dict["our"]["aps"] > best_ds_val_aps:
            best_ds_val_aps = metrics_dict["our"]["aps"]

        if metrics_dict["our"]["aucs"] > best_ds_val_aucs:
            best_ds_val_aucs = metrics_dict["our"]["aucs"]

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss

            best_model = []
            best_model.append(memory)
            best_model.append(gnn)
            best_model.append(embd_to_score_dst)
            best_model.append(embd_to_score_src)
            best_model.append(feats_model)
            best_model.append(embd_to_h0)

            dataset_ = get_config("data_name")
            config_dict = {
                "num_neighbors": hyperparams["num_neighbors"],
                "batch_size": hyperparams["batch_size"],
                "memory_dim": hyperparams["memory_dim"],
                "time_dim": hyperparams["time_dim"],
                "embedding_dim": hyperparams["embedding_dim"],
                "feats_model_h_dim": hyperparams["feats_model_h_dim"],
                "n_comp": hyperparams["n_comp"],
                "lr": hyperparams["lr"],
                "eps": hyperparams["eps"],
                "threshold_eps": hyperparams["threshold_eps"],
                "min_eps": min_eps,
                "threshold_epochs": hyperparams["threshold_epochs"],
                "seed": hyperparams["seed"],
                "data": data,
                "min_dst_idx": min_dst_idx,
                "max_dst_idx": max_dst_idx,
                "all_src": all_src,
                "all_dst": all_dst,
                "rand_all_src": rand_all_src,
                "rand_smpsrc": rand_smpsrc,
                "tmap_src": tmap_src,
                "n_sampled_src": n_sampled_src,
                "rand_all_dst": rand_all_dst,
                "n_sampled_dst": n_sampled_dst,
                "rand_smpdst": rand_smpdst,
                "tmap": tmap,
                "assoc": assoc,
                "val_data": test_data,
                "loader": test_loader,
                "col2K": col2K,
                "best_val_loss_epoch": epoch,
            }
            save_model(
                best_model,
                f"./saved_models/val_loss/{dataset_}_{name}_{now}.pt",
                config_dict,
                neighbor_loader,
            )

        results.append(
            {
                "mean_train_loss": mean_loss,
                "mean_train_loss_feats": mean_loss_feats,
                "mean_val_loss": mean_val_loss,
                "mean_val_loss_feats": mean_val_loss_feats,
                "val_aps_ds": metrics_dict["our"]["aps"],
                "val_aucs_ds": metrics_dict["our"]["aucs"],
                "nn_val_aps_ds": metrics_dict["our"]["nn_aps"],
                "nn_val_aucs_ds": metrics_dict["our"]["nn_aucs"],
                "best_ds_val_aps": best_ds_val_aps,
            }
        )

        results_feats.append([mean_loss_feats])

        if np.abs(eps - min_eps) > 1e-15:
            if tot_epochs > threshold_epochs and results_feats[
                tot_epochs - threshold_epochs
            ][0] - results_feats[-1][0] < threshold_eps * torch.abs(
                results_feats[tot_epochs - threshold_epochs][0]
            ):
                # rescale eps
                eps = eps * 0.1
                print(
                    f"[epoch {epoch} of {num_epochs}]: \t ====== rescaling eps to {eps} "
                    + f"({results_feats[tot_epochs - threshold_epochs][0]} - {results_feats[-1][0]}) ======"
                )

        print(
            f"Epoch: {epoch:02d}:\t Train Loss: {mean_loss:.4f}\t Loss features: {mean_loss_feats:.4f}"
        )
        metrics_str = ", ".join(
            f"{k}: " + " | ".join(f"{m}: {v:.4f}" for m, v in metric2val.items())
            for k, metric2val in metrics_dict.items()
        )
        print(
            f"\t\t Val Loss: {mean_val_loss:.4f}\t Loss features: {mean_val_loss_feats:.4f}\t "
            + "Val metrics: "
            + metrics_str
        )

    # Training has finished, we load the best model, and we want to backup its current
    # memory (which has seen validation edges) so that it can also be used when testing on unseen
    # nodes
    (
        config_dict,
        memory,
        gnn,
        embd_to_score_dst,
        embd_to_score_src,
        feats_model,
        embd_to_h0,
        neighbor_loader,
    ) = load_model(f"./saved_models/val_loss/{dataset_}_{name}_{now}.pt")

    # initialize test negative edge samplers
    convert_data = utils.FromDGBtoPYG(full_data, data)
    test_rand_samplers = {}
    nn_test_rand_samplers = {}
    for NEG_SAMPLE in ["rnd", "hist_nre", "induc_nre"]:
        (
            test_rand_sampler,
            nn_test_rand_sampler,
        ) = utils.initialize_test_negative_edge_sampler(
            NEG_SAMPLE, full_data, val_data, new_node_test_data
        )
        test_rand_sampler.sample = convert_data.sampler_converter_decorator()(
            test_rand_sampler.sample
        )
        nn_test_rand_sampler.sample = convert_data.sampler_converter_decorator()(
            nn_test_rand_sampler.sample
        )
        test_rand_samplers[NEG_SAMPLE] = test_rand_sampler
        nn_test_rand_samplers[NEG_SAMPLE] = nn_test_rand_sampler

    # Test
    with torch.no_grad():
        (
            mean_test_loss,
            mean_test_loss_feats,
            best_test_model_,
            best_test_loss_,
            metrics_dict,
        ) = validate(
            embd_to_score_src,
            embd_to_score_dst,
            memory,
            gnn,
            feats_model,
            embd_to_h0,
            neighbor_loader,
            min_dst_idx,
            max_dst_idx,
            all_src,
            all_dst,
            rand_all_src,
            rand_smpsrc,
            tmap_src,
            n_sampled_src,
            rand_all_dst,
            n_sampled_dst,
            rand_smpdst,
            tmap,
            assoc,
            data,
            eps,
            loader=test_loader,
            compute_loss=True,
            test_rand_samplers=test_rand_samplers,
            nn_test_rand_samplers=nn_test_rand_samplers,
        )

    metrics_str = "\n".join(
        "{}{}".format(k, "\t" if len(k) > 4 else "\t\t")
        + " \t ".join(f"{m}: {v:.4f}" for m, v in metric2val.items())
        for k, metric2val in metrics_dict.items()
    )
    print(
        "=" * 24
        + "\n"
        + f"\t\t Test Loss: {mean_test_loss:.4f}\t Loss features: {mean_test_loss_feats:.4f}\t "
        + "Test metrics: \n"
        + metrics_str
    )
    # save results
    pd.DataFrame(results).to_csv(
        f"./results/results_{dataset_}_{name}_{now}.csv", index=False
    )
    pd.DataFrame(metrics_dict).to_csv(
        f"./results/metrics_{dataset_}_{name}_{now}.csv", index=True
    )


if __name__ == "__main__":
    assert sys.argv[1] in [
        "config_bikeshare.json",
        "config_lastfm.json",
        "config_mooc.json",
        "config_reddit.json",
        "config_wiki.json",
    ], "Config file is invalid."
    main(config_path=f"./configs/{sys.argv[1]}")
