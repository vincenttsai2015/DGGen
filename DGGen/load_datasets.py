import torch
from torch_geometric.datasets import JODIEDataset
import pandas as pd
import numpy as np
import types
import os, sys
import yaml

import utils as gt_utils

# DGB
import sys

sys.path.insert(0, "../")
from DGB.tgn.utils.data_processing import get_data


MAX_FEATURE_COLUMNS = 12


def get_real_data_df(dataset_name, jodie_data_dir):
    if dataset_name in ["wiki", "reddit", "mooc", "lastfm"]:
        dataset = JODIEDataset(
            jodie_data_dir, name="wikipedia" if dataset_name == "wiki" else dataset_name
        )
        data = dataset[0]
        data.msg = data.msg[:, :MAX_FEATURE_COLUMNS]
        data.edge_id = torch.arange(data.num_edges, dtype=torch.long)
    else:
        print("for sure bike data!!")
        data = bs.get_cbs_data(path=None)

    df = (
        pd.DataFrame(
            np.stack(
                [data.src.cpu().numpy(), data.dst.cpu().numpy(), data.t.cpu().numpy()]
                + list(data.msg.cpu().numpy().T),
                1,
            )
        )
        .astype({0: int, 1: int, 2: int})
        .rename(columns={0: "i", 1: "j", 2: "t"})
    )
    return df


def get_real_test_data_df(
    dataset_name,
    jodie_data_dir="./data/",
    cbs_data_file="./data/202207-capitalbikeshare-tripdata.csv",
    tgn_data_dir="../DGB/DGB/tgn/",
):
    # load data in DGB format
    ### Extract data for training, validation and testing
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
        "wikipedia" if dataset_name == "wiki" else dataset_name,
        val_ratio,
        test_ratio,
        different_new_nodes_between_val_and_test=different_new_nodes,
        randomize_features=randomize_features,
        path=tgn_data_dir,
    )

    # load data in TemporalData format
    if dataset_name in ["wiki", "wikipedia", "reddit", "mooc", "lastfm"]:
        dataset = JODIEDataset(
            jodie_data_dir, name="wikipedia" if dataset_name == "wiki" else dataset_name
        )
        data = dataset[0]
        data.edge_id = torch.arange(data.num_edges, dtype=torch.long)
        data = data.to(torch.device("cpu"))
        # reduce the number of features
        data.msg = data.msg[:, :MAX_FEATURE_COLUMNS]
        col2K = {i + 1: "gmm" for i in range(data.msg.shape[1])}
    else:
        print("for sure bike data!!")
        from bikeshare import get_cbs_data

        data, col2K = get_cbs_data(path=cbs_data_file)
        data.edge_id = torch.arange(data.num_edges, dtype=torch.long)

    # convert DBG test data to TemporalData format
    test_temp_data = gt_utils.convert_dgb_data_to_pyg_TemporalData(
        test_data, min_dgb_edge_idx=full_data.edge_idxs.min(), pyg_temporaldata=data
    )

    df = (
        pd.DataFrame(
            np.stack(
                [
                    test_temp_data.src.cpu().numpy(),
                    test_temp_data.dst.cpu().numpy(),
                    test_temp_data.t.cpu().numpy(),
                ]
                + list(test_temp_data.msg[:, :MAX_FEATURE_COLUMNS].cpu().numpy().T),
                1,
            )
        )
        .astype({0: int, 1: int, 2: int})
        .rename(columns={0: "i", 1: "j", 2: "t"})
    )
    # shift time col
    df["t"] -= df["t"].min()
    return df


def load_old_dggen_gen_data(fname):
    gen_data = torch.load(fname, map_location=torch.device("cpu"))

    st = 0
    en = None
    # excludes the dt (time difference) column, if present
    if torch.allclose(gen_data.t[1:] - gen_data.t[:-1], gen_data.msg[1:, 0].long()):
        dt_col = 1
    else:
        dt_col = 0
    df0 = (
        pd.DataFrame(
            np.stack(
                [
                    gen_data.src.numpy()[st:en],
                    gen_data.dst.numpy()[st:en],
                    gen_data.t.numpy()[st:en],
                ]
                + list(
                    gen_data.msg[:, dt_col : MAX_FEATURE_COLUMNS + dt_col]
                    .numpy()[st:en]
                    .T
                ),
                1,
            )
        )
        .astype({0: int, 1: int, 2: int})
        .rename(columns={0: "i", 1: "j", 2: "t"})
    )
    return df0


def get_tigger_edge_list_to_csv(dataset_name, tigger_data_dir):
    """
    preprocess TIGGER data
    dataset_names = ['cbs', 'wiki', 'reddit', 'lastfm', 'mooc']
    """
    try:
        df_tigger = pd.read_csv(
            tigger_data_dir + f"{dataset_name}_tigger_edge_list.csv"
        )
    except FileNotFoundError:
        fname = f"{tigger_data_dir}{dataset_name}_tigger_data"
        gen_data = torch.load(fname, map_location=torch.device("cpu"))

        columns = ["i", "j", "t"]
        dtypes = [int, int, int]

        df_tigger = pd.DataFrame(
            np.stack(
                [
                    gen_data.src.cpu().numpy(),
                    gen_data.dst.cpu().numpy(),
                    gen_data.t.cpu().numpy(),
                ],
                1,
            ),
            columns=columns,
        ).astype(dict(zip(columns, dtypes)))

        df_tigger.to_csv(
            f"{tigger_data_dir}{dataset_name}_tigger_edge_list.csv", index=False
        )
    return df_tigger


def load_test_data(
    dataset_name,
    cfg,
):
    # load data
    df_list, names_list = [], []

    if isinstance(dataset_name, str):
        dataset_name_real, dataset_name_tigger, dataset_name_dggen = (
            dataset_name,
            dataset_name,
            dataset_name,
        )
    elif len(dataset_name) == 3:
        dataset_name_real, dataset_name_tigger, dataset_name_dggen = dataset_name
    else:
        sys.exit("process_dataset: dataset_name has a wrong format.")

    # load real data
    df_real = get_real_test_data_df(
        dataset_name_real,
        jodie_data_dir=cfg.jodie_data_dir,
        cbs_data_file=cfg.cbs_data_file,
        tgn_data_dir=cfg.tgn_data_dir,
    )
    df_list.append(df_real)
    names_list.append("real")

    # load TIGGER data
    print(
        f"Loading TIGGER file from "
        + cfg.tigger_data_dir
        + f"{dataset_name_tigger}_tigger_edge_list.csv"
    )
    df_tigger = get_tigger_edge_list_to_csv(dataset_name_tigger, cfg.tigger_data_dir)
    df_list.append(df_tigger)
    names_list.append("tigger")

    # load dggen data
    # load dggen data
    dggen_files = [
        f
        for f in os.listdir(cfg.dggen_dir)
        if dataset_name_dggen in f and f.endswith(".pt")
    ]

    if dggen_files:
        dggen_fname = f"{cfg.dggen_dir}{dggen_files[0]}"
        print(f"Loading TG-Gen file from {dggen_fname}")
        df_dggen = load_old_dggen_gen_data(dggen_fname)
    else:
        print("No file found that matches the conditions.")
        exit(1)
    df_list.append(df_dggen)
    names_list.append("dggen")

    assert (
        names_list[0] == "real"
    ), "real should be the first name and real data should be the first df"
    return df_list, names_list


def main(cfg, save_each_dataset=False):
    datasets = ["cbs", "wiki", "reddit", "mooc", "lastfm"]
    dataset2model2metric2val = {}
    for dataset_name in datasets:
        _dataset_name = dataset_name
        print("=" * 32 + f"  Processing {dataset_name} dataset...")
        df_list, names_list = load_test_data(_dataset_name, cfg)
    print(f"dataset names {names_list}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            config_dict = yaml.safe_load(f)
    else:
        config_dict = {
            "dggen_dir": "./saved_models/val_loss/",
            "tigger_data_dir": "./synthetic_data/tigger/",
            "jodie_data_dir": "./data/",
            "cbs_data_file": "./data/202207-capitalbikeshare-tripdata.csv",
            "tgn_data_dir": "./DGB/DGB/tgn/",
        }
    cfg = types.SimpleNamespace(**config_dict)

    main(cfg, save_each_dataset=False)
