import numpy as np
import torch
from utils import subsample_src, subsample_dst, preprocess
import tqdm


def train_epoch(
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
    train_data,
    max_dst_idx,
):
    memory.train()
    gnn.train()
    embd_to_score_src.train()
    embd_to_score_dst.train()
    feats_model.train()
    embd_to_h0.train()

    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    total_loss = 0
    total_loss_feats = 0.0
    total_norm_feats = 0.0

    for batch in tqdm.tqdm(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()

        loss_unnormalized = torch.tensor([0.0])
        norm = torch.tensor(1.0)

        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        neg_dst = torch.randint(
            min_dst_idx,
            max_dst_idx + 1,
            (src.size(0),),
            dtype=torch.long,
            device=device,
        )

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

        n_id = torch.cat(
            [src, pos_dst, neg_dst, sampled_src.unique(), sampled_dst.unique()]
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

        loss_feats = -sum(
            (
                c.ptdist(x).log_prob(targets[:, c.mask, -1]).sum()
                if c.distrib
                is torch.distributions.mixture_same_family.MixtureSameFamily
                else c.ptdist(x).log_prob(targets[:, c.mask]).sum()
            )
            for c in feats_model.columns
        )

        loss_unnormalized += loss_feats.cpu()  # type: ignore
        norm += np.prod(targets.shape)
        total_loss_feats += loss_feats
        total_norm_feats += np.prod(targets.shape)

        # using total losses (sum) and normalizing here
        loss = loss_unnormalized / norm

        loss.backward()
        optimizer.step()

        # save best model
        if loss.item() < best_loss:
            best_loss = loss.item()

        # FIXME
        best_model = None

        # Update memory and neighbor loader with ground-truth state.
        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)
        memory.detach()
        total_loss += float(loss) * batch.num_events

    return (
        total_loss / train_data.num_events,
        total_loss_feats / total_norm_feats,
        best_model,
        best_loss,
    )
