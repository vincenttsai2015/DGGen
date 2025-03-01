import torch
import tqdm


def update_embeddings(
    n_id, gen_data, neighbor_loader, best_memory, best_gnn, embeddings, device="cuda"
):
    n_id, edge_index, e_id = neighbor_loader(n_id)
    # Get updated memory of all nodes involved in the computation.
    z, last_update = best_memory(n_id)
    # compute embeddings
    new_embeddings = best_gnn(
        z,
        last_update,
        edge_index,
        gen_data.t[e_id].to(device),
        gen_data.msg[e_id].to(device),
    )
    # update embeddings
    embeddings[n_id] = new_embeddings


class GeneratedData:
    def __init__(self, size, msg_dim, device="cuda"):
        self.src = torch.empty([size]).long().to(device)
        self.dst = torch.empty([size]).long().to(device)
        self.t = torch.empty([size]).long().to(device)
        self.msg = torch.empty([size, msg_dim]).float().to(device)


@torch.no_grad()
def generate(
    best_memory,
    best_gnn,
    best_embd_to_score_src,
    best_embd_to_score_dst,
    best_feats_model,
    best_embd_to_h0,
    neighbor_loader,
    all_n_id,
    embeddings,
    seed,
    all_src,
    all_dst,
    min_dst_idx,
    preprocess=None,
    batch_size=10,
    num_batches=100,
    empty_memory_and_graph=True,
    initial_time=0,
    device="cuda",
):
    best_memory.eval()
    best_gnn.eval()
    best_embd_to_score_src.eval()
    best_embd_to_score_dst.eval()
    best_feats_model.eval()
    best_embd_to_h0.eval()

    gen_data = GeneratedData(
        batch_size * num_batches, best_memory.raw_msg_dim, device=device
    )

    torch.manual_seed(seed)  # Ensure deterministic sampling across epochs.

    if empty_memory_and_graph:
        best_memory.reset_state()  # Start with a fresh memory.
        neighbor_loader.reset_state()  # Start with an empty graph.
        update_embeddings(
            all_n_id,
            gen_data,
            neighbor_loader,
            best_memory,
            best_gnn,
            embeddings,
            device=device,
        )  # Compute initial embeddings for all nodes

    for batch_idx in tqdm.tqdm(range(num_batches)):

        # sample src
        scores = best_embd_to_score_src(embeddings[all_src])
        distr = torch.distributions.Categorical(logits=scores.squeeze())
        src = distr.sample([batch_size])

        # sample dst
        scores = best_embd_to_score_dst(embeddings[src], embeddings[all_dst])
        distr = torch.distributions.Categorical(logits=scores)
        pos_dst = distr.sample() + min_dst_idx

        # sample time + features
        h_0 = best_embd_to_h0(embeddings[src], embeddings[pos_dst]).unsqueeze(0)
        t_msg = best_feats_model.sample(h_0.shape[1], preprocess, hx=h_0, device=device)
        t = t_msg[:, 0].long().cumsum(0) + initial_time
        msg = t_msg

        # add generated data to `gen_data`
        i_s, i_e = batch_idx * batch_size, (batch_idx + 1) * batch_size
        gen_data.src[i_s:i_e] = src
        gen_data.dst[i_s:i_e] = pos_dst
        gen_data.t[i_s:i_e] = t
        gen_data.msg[i_s:i_e] = msg

        initial_time = t[-1]

        # Update memory and neighbor loader with ground-truth state.
        best_memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

        # get ids of nodes involved
        n_id = torch.cat([src, pos_dst]).unique()
        # update embeddings
        update_embeddings(
            n_id,
            gen_data,
            neighbor_loader,
            best_memory,
            best_gnn,
            embeddings,
            device=device,
        )

    gen_data.msg = gen_data.msg[:, 1:]
    return gen_data
