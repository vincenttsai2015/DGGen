import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from utils import subsample_src, subsample_dst, preprocess
import tqdm


@torch.no_grad()
def validate(
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
    loader,
    compute_loss=True,
    test_rand_samplers=None,
    nn_test_rand_samplers=None,
):
    memory.eval()
    gnn.eval()
    embd_to_score_src.eval()
    embd_to_score_dst.eval()
    feats_model.eval()
    embd_to_h0.eval()

    total_loss = 0
    total_loss_feats = 0.0
    total_norm_feats = 0.0
    metrics_dict = {}
    metrics_dict["our"] = {
        "aps": [],
        "aucs": [],
        "nn_aps": [],
        "nn_aucs": [],
    }
    pos_probs_neg_probs = {}
    pos_probs_neg_probs["our"] = {
        "pos_probs": [],
        "neg_probs": [],
    }

    if test_rand_samplers is not None:
        for k in test_rand_samplers.keys():
            metrics_dict[k] = {
                "aps": [],
                "aucs": [],
                "nn_aps": [],
                "nn_aucs": [],
            }
            pos_probs_neg_probs[k] = {
                "pos_probs": [],
                "neg_probs": [],
            }
    dgb_sampled_src_dst = {}
    nn_dgb_sampled_src_dst = {}
    device = assoc.device

    for batch in tqdm.tqdm(loader):
        batch = batch.to(device)

        loss_unnormalized = torch.tensor([0.0], device=device)
        norm = torch.tensor(1.0, device=device)

        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        neg_dst = torch.randint(
            min_dst_idx,
            max_dst_idx + 1,
            (src.size(0),),
            dtype=torch.long,
            device=device,
        )

        if compute_loss:
            # Sample a subset of origins, including the positive ones
            sampled_src, idx_pos_src = subsample_src(
                src,
                all_src,
                rand_all_src,
                rand_smpsrc,
                tmap_src,
                n_sampled_src,
                device,
                sample_all=False,
            )

            # Sample a subset of destinations, including the positive ones
            sampled_dst, idx_pos_dst = subsample_dst(
                pos_dst,
                rand_all_dst,
                n_sampled_dst,
                rand_smpdst,
                all_dst,
                min_dst_idx,
                tmap,
                device,
                sample_all=False,
            )
        else:
            sampled_src = torch.tensor([], dtype=torch.long).to(device)
            sampled_dst = torch.tensor([], dtype=torch.long).to(device)

        # dgb neg samples
        if test_rand_samplers is not None:
            dgb_nodes = []
            for k, sampler in test_rand_samplers.items():
                dgb_sampled_src_dst[k] = sampler.sample(
                    src.size(0), batch.t[0].cpu().numpy(), batch.t[-1].cpu().numpy()
                )
                dgb_nodes += dgb_sampled_src_dst[k]
            for k, sampler in nn_test_rand_samplers.items():
                nn_dgb_sampled_src_dst[k] = sampler.sample(
                    batch.new_node.sum().item(),
                    batch.t[0].cpu().numpy(),
                    batch.t[-1].cpu().numpy(),
                )
                dgb_nodes += nn_dgb_sampled_src_dst[k]
            dgb_nodes = torch.cat([all_src, all_dst])  # all nodes
        else:
            dgb_nodes = torch.tensor([], dtype=torch.long).to(device)

        n_id = torch.cat(
            [
                src,
                pos_dst,
                neg_dst,
                sampled_src.unique(),
                sampled_dst.unique(),
                dgb_nodes,
            ]
        ).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Get updated memory of all nodes involved in the computation.
        z, last_update = memory(n_id)
        # compute embeddings
        z = gnn(
            z,
            last_update,
            edge_index,
            data.t[e_id].to(device),
            data.msg[e_id].to(device),
        )

        if compute_loss:
            # Origins
            scores_src = embd_to_score_src(z[assoc[sampled_src]])
            # transform scores into log probabilities
            log_probs_src = scores_src - scores_src.logsumexp(dim=0).unsqueeze(1)
            # get the log probability of the positive targets only
            log_probs_pos_src = torch.gather(
                log_probs_src, dim=0, index=idx_pos_src.unsqueeze(dim=1)
            )
            loss_src = -1 * log_probs_pos_src.sum()
            loss_unnormalized += loss_src.cpu()
            norm += np.prod(log_probs_pos_src.shape)

            # Destinations
            scores_dst = embd_to_score_dst(z[assoc[src]], z[assoc[sampled_dst]])
            # transform scores into log probabilities
            log_probs_dst = scores_dst - scores_dst.logsumexp(dim=1).unsqueeze(1)
            # get the log probability of the positive targets only
            log_probs_pos_dst = torch.gather(
                log_probs_dst, dim=1, index=idx_pos_dst.unsqueeze(dim=1)
            )
            loss_dst = -1 * log_probs_pos_dst.sum()
            loss_unnormalized += loss_dst.cpu()
            norm += np.prod(log_probs_pos_dst.shape)

            # add noise
            with torch.no_grad():
                noise = torch.randn_like(msg) * eps
                noise[:, 0] = 0
                noise[:, feats_model.categorical_idx] = 0
                msg = msg + noise
            msgp = preprocess(msg, device)
            inputs = msgp[:, :-1]
            targets = msgp[:, 1:]
            h_0 = embd_to_h0(z[assoc[src]], z[assoc[pos_dst]]).unsqueeze(0)
            x = feats_model(inputs, h_0)
            x = torch.nan_to_num(x, nan=0.01)

            loss_feats = -sum(
                (
                    c.ptdist(x)
                    .log_prob(torch.nan_to_num(targets[:, c.mask, -1], nan=0.01))
                    .sum()
                    if c.distrib
                    is torch.distributions.mixture_same_family.MixtureSameFamily
                    else c.ptdist(x).log_prob(targets[:, c.mask]).sum()
                )
                for c in feats_model.columns
            )

            loss_unnormalized += loss_feats.cpu()
            norm += np.prod(targets.shape)
            total_loss_feats += loss_feats
            total_norm_feats += np.prod(targets.shape)

            # using total losses (sum) and normalizing here
            loss = loss_unnormalized / norm

            # FIXME
            best_loss = None
            best_model = None

        # link pred metrics
        z_dst = torch.stack([z[assoc[pos_dst]], z[assoc[neg_dst]]], dim=1)
        scores_dst = embd_to_score_dst(z[assoc[src]], z_dst)
        # transform scores into probabilities
        probs_dst = (scores_dst - scores_dst.logsumexp(dim=1).unsqueeze(1)).exp()
        # get the probability of the positive and negative targets
        pos_out = probs_dst[:, 0]
        neg_out = probs_dst[:, 1]
        y_pred = torch.cat([pos_out, neg_out], dim=0).cpu()
        y_true = torch.cat(
            [torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))], dim=0
        )
        metrics_dict["our"]["aps"].append(average_precision_score(y_true, y_pred))
        metrics_dict["our"]["aucs"].append(roc_auc_score(y_true, y_pred))

        if "new_node" in batch._store:
            # link pred metrics for new nodes
            # select the interactions with new nodes
            probs_dst = probs_dst[batch.new_node.bool()]
            # get the probability of the positive and negative targets
            pos_out = probs_dst[:, 0]
            neg_out = probs_dst[:, 1]
            y_pred = torch.cat([pos_out, neg_out], dim=0).cpu()
            y_true = torch.cat(
                [torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))], dim=0
            )
            pos_probs_neg_probs["our"]["pos_probs"].append(pos_out)
            pos_probs_neg_probs["our"]["neg_probs"].append(neg_out)

        if test_rand_samplers is not None:
            #  Origins
            scores_src = embd_to_score_src(z[assoc[all_src]])
            # transform scores into log probabilities
            log_probs_src = scores_src - scores_src.logsumexp(dim=0).unsqueeze(1)
            # Destinations
            scores_dst = embd_to_score_dst(z[assoc[all_src]], z[assoc[all_dst]])
            # transform scores into log probabilities
            log_probs_dst = scores_dst - scores_dst.logsumexp(dim=1).unsqueeze(1)
            # joint probabilities
            joint_probs = (log_probs_src + log_probs_dst).exp()
            # positive edges probs
            pos_probs = joint_probs[src, pos_dst - min_dst_idx]

            for k, (neg_src, neg_dst) in dgb_sampled_src_dst.items():
                # negative edges probs
                neg_probs = joint_probs[neg_src, neg_dst - min_dst_idx]
                y_pred = torch.cat([pos_probs, neg_probs], dim=0).cpu().numpy()
                y_true = (
                    torch.cat(
                        [torch.ones_like(pos_probs), torch.zeros_like(neg_probs)], dim=0
                    )
                    .cpu()
                    .numpy()
                )
                metrics_dict[k]["aps"].append(average_precision_score(y_true, y_pred))
                metrics_dict[k]["aucs"].append(roc_auc_score(y_true, y_pred))

            for k, (neg_src, neg_dst) in nn_dgb_sampled_src_dst.items():
                # positive edges probs
                pos_probs = joint_probs[
                    src[batch.new_node.bool()],
                    pos_dst[batch.new_node.bool()] - min_dst_idx,
                ]
                # negative edges probs
                neg_probs = joint_probs[neg_src, neg_dst - min_dst_idx]
                assert (
                    pos_probs.shape == neg_probs.shape
                ), "pos and neg shapes do not match"
                pos_probs_neg_probs[k]["pos_probs"].append(pos_probs)
                pos_probs_neg_probs[k]["neg_probs"].append(neg_probs)

        # Update memory and neighbor loader with ground-truth state.
        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

        total_loss += float(loss) * batch.num_events

    # compute metrics for new nodes over batches and then average
    for k, pos_neg in pos_probs_neg_probs.items():
        if k == "our" or test_rand_samplers is not None:
            all_pos_probs = torch.cat(pos_neg["pos_probs"])
            all_neg_probs = torch.cat(pos_neg["neg_probs"])
            batch_size = loader.batch_sampler.sampler.data_source.step
            for i in range(0, len(all_pos_probs) - batch_size + 1, batch_size):
                pos_probs = all_pos_probs[i : i + batch_size]
                neg_probs = all_neg_probs[i : i + batch_size]
                y_pred = torch.cat([pos_probs, neg_probs], dim=0).cpu().numpy()
                y_true = (
                    torch.cat(
                        [torch.ones_like(pos_probs), torch.zeros_like(neg_probs)], dim=0
                    )
                    .cpu()
                    .numpy()
                )
                metrics_dict[k]["nn_aps"].append(
                    average_precision_score(y_true, y_pred)
                )
                metrics_dict[k]["nn_aucs"].append(roc_auc_score(y_true, y_pred))
    metrics_dict = {
        k: {m: float(torch.tensor(v).mean()) for m, v in metric2val.items()}
        for k, metric2val in metrics_dict.items()
    }
    return (
        total_loss / loader.data.num_events,
        total_loss_feats / total_norm_feats,
        best_model,
        best_loss,
        metrics_dict,
    )
