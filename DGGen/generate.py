import os, sys
import torch
import glob

import utils
import inference
import model


def get_synthetic_data(saved_model_path, out_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_fname = "generated_" + saved_model_path.split("/")[-1]
    OUT_PATH = out_path + out_fname

    print("loading model...")
    (
        config_dict,
        memory,
        gnn,
        embd_to_score_dst,
        embd_to_score_src,
        feats_model,
        embd_to_h0,
        neighbor_loader,
    ) = utils.load_model(saved_model_path)

    cfg = utils.SimpleNamespace(**config_dict)
    data = cfg.data

    BATCH_SIZE = cfg.batch_size
    val_ratio = 0.15
    test_ratio = 0.15
    num_interactions_to_generate = int(
        len(cfg.data) * test_ratio
    )  # size of the test dataset
    NUM_BATCHES = num_interactions_to_generate // BATCH_SIZE

    print(f"generating {NUM_BATCHES} batches of size {BATCH_SIZE} ...")
    all_n_id = torch.arange(data.num_nodes).long().to(device)
    embeddings = torch.empty([data.num_nodes, cfg.embedding_dim]).to(device)

    data_synthetic = inference.generate(
        best_memory=memory,
        best_gnn=gnn,
        best_embd_to_score_src=embd_to_score_src,
        best_embd_to_score_dst=embd_to_score_dst,
        best_feats_model=feats_model,
        best_embd_to_h0=embd_to_h0,
        neighbor_loader=neighbor_loader,
        all_n_id=all_n_id,
        embeddings=embeddings,
        seed=cfg.seed,
        all_src=cfg.all_src,
        all_dst=cfg.all_dst,
        min_dst_idx=cfg.min_dst_idx,
        preprocess=model.preprocess,
        batch_size=BATCH_SIZE,
        num_batches=NUM_BATCHES,
        device=device,
    )

    print("saving...")
    data_synthetic = utils.TemporalData(
        src=data_synthetic.src,
        dst=data_synthetic.dst,
        t=data_synthetic.t,
        msg=data_synthetic.msg,
    )
    torch.save(data_synthetic, OUT_PATH)

    df = utils.temporaldata_to_df(data_synthetic)
    df.to_csv(OUT_PATH[: OUT_PATH.rindex(os.path.extsep)] + ".csv", index=False)
    print("done.")


if __name__ == "__main__":
    assert sys.argv[1] in [
        "bikeshare",
        "wikipedia",
        "reddit",
        "mooc",
        "lastfm",
    ], "Provide valid dataset for synthetic data generation."
    out_path = f"./results/synthetic_data/{sys.argv[1]}"
    os.makedirs("./results/synthetic_data/", exist_ok=True)

    # If using train.py, do not need to change hard code
    prepend = "cbs" if sys.argv[1] == "bikeshare" else sys.argv[1]
    model_dir = "./saved_models/val_loss"

    # Find the first file that starts with the prepend string
    try:
        matching_files = glob.glob(os.path.join(model_dir, f"{prepend}*"))
    except Exception:
        raise FileNotFoundError(
            "Make sure to run train.py first on the dataset you wish to generate from. See README for more details."
        )
    if matching_files:
        saved_model_path = matching_files[0]
    else:
        raise FileNotFoundError(
            "Make sure to run train.py first on the dataset you wish to generate from. See README for more details."
        )
    get_synthetic_data(saved_model_path, out_path)
