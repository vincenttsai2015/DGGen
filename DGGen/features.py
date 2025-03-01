import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics
import torch
from scipy.spatial.distance import jensenshannon

import yaml
import types
import sys
import load_datasets as ld
import pathlib

src_col = "i"
dst_col = "j"
t_col = "t"


def k_div(p0, q0):
    p = p0[p0 > 0]
    q = q0[p0 > 0]
    return np.sum(p * np.log2(2 * p / (p + q)))


def weighted_jaccard_distance(x, y):
    assert (
        len(np.asarray(x).shape) == 1 and len(np.asarray(y).shape) == 1
    ), "weighted_jaccard_distance: arguments must be one-dimensional arrays."
    q = np.stack([x, y])
    return 1.0 - np.sum(np.amin(q, axis=0)) / np.sum(np.amax(q, axis=0))


def get_metrics(a, a0, n_bins=30, rounding=2, logbin=False):
    bmin, bmax = pd.concat([a, a0]).min(), pd.concat([a, a0]).max()
    if bmax > 0.0:
        brange = np.log10(bmax - bmin)
    else:
        brange = -1
    if (logbin or brange > 4) and bmin >= 0.0:
        bins = np.logspace(np.log10(bmin + 1e-1), np.log10(bmax), n_bins)
    else:
        bins = np.linspace(bmin, bmax, n_bins)
    bc0, be0 = np.histogram(a0, density=True, bins=bins)
    bc, be = np.histogram(a, density=True, bins=bins)

    rmse = sklearn.metrics.mean_squared_error(bc, bc0) ** 0.5
    js_dist = jensenshannon(bc, bc0)
    jac_dist = weighted_jaccard_distance(bc, bc0)
    metrics = {
        "J-S dist": js_dist,
        "Jac dist": jac_dist,
    }
    metrics_str = ", ".join(
        [": ".join([k, f"{v:.{rounding}f}"]) for k, v in metrics.items()]
    )
    return metrics, metrics_str, bins


def plot_train_val_losses(
    results,
    offset=0.0,
    rescale_factor=1.0,
    every=1,
):
    fig = plt.figure(figsize=(16, 8))

    fig.add_subplot(221)
    plt.plot(
        np.arange(1, len(results) + 1)[::every],
        np.array(results)[::every, 0] + offset,
        "-",
    )
    plt.ylabel("Train loss")
    plt.xlabel("Epoch")

    fig.add_subplot(223)
    plt.plot(
        np.arange(1, len(results) + 1)[::every],
        rescale_factor * np.array(results)[::every, 1] + offset,
        "-",
        color="orange",
    )
    plt.ylabel("Val loss")
    plt.xlabel("Epoch")

    fig.add_subplot(222)
    plt.plot(
        np.arange(1, len(results) + 1)[::every],
        rescale_factor * np.array(results)[::every, 2] + offset,
        "-",
        color="red",
    )
    plt.ylabel("Val AP")
    plt.xlabel("Epoch")

    fig.add_subplot(224)
    plt.plot(
        np.arange(1, len(results) + 1)[::every],
        rescale_factor * np.array(results)[::every, 3] + offset,
        "-",
        color="green",
    )
    plt.ylabel("Val AUC")
    plt.xlabel("Epoch")
    return plt.gca()


def plot_feature_hist(
    df,
    df0,
    column,
    show_title=False,
    n_bins=30,
    kwargs={
        "alpha": 0.5,
    },
    ax=None,
):
    a = df[column]
    a0 = df0[column]

    metrics, metrics_str, bins = get_metrics(a, a0, n_bins=n_bins)

    ax = a.plot(kind="hist", density=True, bins=bins, ax=ax, **kwargs)
    ax = a0.plot(kind="hist", density=True, bins=bins, ax=ax, **kwargs)

    if show_title:
        ax.axes.set_title(f"{metrics_str}")
    ax.set_xlabel(f"Feature: {column}")
    return ax, metrics


def scatter_plot_feats(
    df,
    df0,
    columns,
    n=100,
    noise=1e-1,
    kwargs={"alpha": 0.5, "marker": ".", "lw": 0, "legend": None},
    ax=None,
):
    i, j = 0, 1
    a = df.sample(n)[columns]
    a0 = df0.sample(n)[columns]

    # add noise
    if noise is not None:
        noise = np.random.randn(*a.shape) * noise
        a = a + noise
        a0 = a0 + noise

    ax = a.plot(*columns, ax=ax, **kwargs)
    ax = a0.plot(*columns, ax=ax, **kwargs)
    ax.set_xlabel(f"Feature: {columns[i]}")
    ax.set_ylabel(f"Feature:\n{columns[j]}")
    return ax


def get_interevent_times(data, minlen=None):
    dt = torch.cat(
        [torch.tensor([0.0]).to(data.t.device), data.t[1:minlen] - data.t[: minlen - 1]]
    )
    return dt


def hist_interevent_time(
    data,
    gen_data,
    show_title=False,
    n_bins=30,
    kwargs={
        "alpha": 0.5,
    },
    ax=None,
):
    # interevent time distribution
    ofst = 0
    minlen = min(len(data), len(gen_data))

    s = pd.Series(ofst + get_interevent_times(data, minlen=minlen).cpu().numpy())
    s0 = pd.Series(ofst + get_interevent_times(gen_data, minlen=minlen).cpu().numpy())

    metrics, metrics_str, bins = get_metrics(s, s0, n_bins=n_bins)

    ax = s.plot(kind="hist", ax=ax, bins=bins, density=True, **kwargs)
    ax = s0.plot(kind="hist", ax=ax, bins=bins, density=True, **kwargs)

    if show_title:
        ax.axes.set_title(f"{metrics_str}")
    ax.set_xlabel("Interevent time [s]")
    ax.set_yscale("log")
    return ax, metrics


def plot_interactions_vs_time(gen_data, data):
    # interaction number (x) vs time in seconds (y)
    plt.plot(data.t.cpu().numpy())
    plt.plot(gen_data.t.cpu().numpy()[: len(data.t)])
    return plt.gca()


def arrange_plots(list_plots, n_cols=3, cols_scale=4, rows_scale=4):
    """
    list_plots: List,
        list_plots = [
            [plot_func, [pd.DataFrame(np.arange(3))]],
            [plot_func, [pd.DataFrame(np.arange(3))]],
        ]
    """
    n_plots = len(list_plots)
    n_cols = min(n_plots, n_cols)
    n_rows = np.ceil(n_plots / n_cols).astype(int)
    fig, axs = plt.subplots(n_rows, n_cols)
    if len(axs.shape) == 1:
        axs = np.expand_dims(axs, 0)
    fig.set_size_inches(n_cols * cols_scale, n_rows * rows_scale)

    for n, [plot_func, plot_args] in enumerate(list_plots):
        i, j = n // n_cols, n % n_cols
        plot_func(*plot_args, ax=axs[i, j])

    for n in range(n_plots, n_cols * n_rows):
        i, j = n // n_cols, n % n_cols
        fig.delaxes(axs[i, j])

    plt.tight_layout()
    return fig, axs


def hist_groupby_col(
    df,
    df0,
    groupby_columns,
    xlabel=None,
    show_title=False,
    n_bins=20,
    kwargs={
        "alpha": 0.5,
    },
    ax=None,
):
    other_col = df0.columns[-1]
    a = df.groupby(groupby_columns).count()[other_col]
    a0 = df0.groupby(groupby_columns).count()[other_col]

    metrics, metrics_str, bins = get_metrics(a, a0, n_bins=n_bins)

    ax = a.plot(kind="hist", density=True, bins=bins, ax=ax, **kwargs)
    ax = a0.plot(kind="hist", density=True, bins=bins, ax=ax, **kwargs)

    ax.axes.set_yscale("log")
    if xlabel is None:
        xlabel = f"{groupby_columns}"
    ax.set_xlabel(xlabel)
    if show_title:
        ax.axes.set_title(f"{metrics_str}")
    return ax, metrics


def get_hourly_counts(times, init_datetime, title="count", start_time_col=t_col):
    times = init_datetime + times.apply(lambda x: pd.Timedelta(x, unit="s"))
    tdf = pd.DataFrame(times)
    tdf[title] = np.ones(len(tdf))
    return tdf.groupby([tdf[start_time_col].dt.hour])[[title]].sum()


def plot_hourly_counts(
    df0, groupby_col, topk, init_datetime, ax0=None, ax1=None, start_time_col=t_col
):
    a = []
    for s in topk.index.values:
        times = df0[df0[groupby_col] == s][start_time_col]
        a += [get_hourly_counts(times, init_datetime, title=s)]

    times = df0[start_time_col]
    a += [
        get_hourly_counts(
            times, init_datetime, title="all", start_time_col=start_time_col
        )
    ]

    fig = plt.figure(figsize=(12, 5))

    fig.add_subplot(121)
    ax = plt.gca()
    for c in a:
        ax = c.plot(ax=ax)
    plt.yscale("log")

    fig.add_subplot(122)
    ax = plt.gca()
    for c in a[:-1]:
        ax = c.plot(ax=ax)
    return ax


def hist_net_interaction_count(
    df,
    df0,
    n_bins=40,
    kwargs={
        "alpha": 0.5,
    },
    show_title=False,
    ax=None,
):
    a_a0 = []
    for _df in [df, df0]:
        diff = pd.merge(
            _df.groupby([src_col])[dst_col].count(),
            _df.groupby([dst_col])[src_col].count(),
            left_index=True,
            right_index=True,
            how="outer",
        ).fillna(0)
        a_a0 += [(diff[dst_col] - diff[src_col])]
    a, a0 = a_a0
    metrics, metrics_str, bins = get_metrics(a, a0, n_bins=n_bins)

    ax = a.hist(bins=bins, density=True, ax=ax, **kwargs)
    ax = a0.hist(bins=bins, density=True, ax=ax, **kwargs)
    ax.axes.set_xlabel("Net difference dst - src count per node")
    if show_title:
        ax.axes.set_title(f"{metrics_str}")
    return ax, metrics


# feature hists


def compute_js_dist_feature(real, synt, col, bins=40):
    xr = real[:, col]
    xs = synt[:, col]
    _bins = np.linspace(min(xr.min(), xs.min()), max(xr.max(), xs.max()), bins)
    h, be = np.histogram(xr, bins=_bins, density=True)
    h_s, be = np.histogram(xs, bins=be, density=True)
    return jensenshannon(h, h_s)


def compute_js_dist_features(real, synt, bins=40):
    return [
        compute_js_dist_feature(real, synt, col, bins=bins)
        for col in range(real.shape[1])
    ]


def plot_feature_hist(real, synt, col, title=None, ax=None, bins=40, alpha=0.5):
    js_dist = f"J-S dist. {compute_js_dist_feature(real, synt, col, bins=bins):.3f}"
    xr = real[:, col]
    xs = synt[:, col]

    def qmin(a):
        return np.quantile(a, 0.02) if title == "MOOC" else a.min()

    def qmax(a):
        return np.quantile(a, 0.98) if title == "MOOC" else a.max()

    _bins = np.linspace(min(qmin(xr), qmin(xs)), max(qmax(xr), qmax(xs)), bins)
    ax = pd.Series(real[:, col]).hist(
        alpha=alpha, ax=ax, bins=_bins, density=True, label="Real"
    )
    ax = pd.Series(synt[:, col]).hist(
        alpha=alpha, ax=ax, bins=_bins, density=True, label="Synthetic\n(DG-Gen)"
    )

    if title is not None:
        ax.set_title(title, fontsize=18)
    ax.set_ylabel("Density")
    ax.set_xlabel(f"Feature {col + 1}")
    ax.legend(fontsize=12)
    ax.legend(title=js_dist)
    if title == "Reddit":
        ax.axis(xmin=-6, xmax=12)
    ax.tick_params(labelsize=6)
    return ax


def plot_feature_hists(real, synt, ax=None, bins=40, alpha=0.5, savefig=None):
    plot_list = [[plot_feature_hist, [real, synt, col]] for col in range(real.shape[1])]
    fig, axs = arrange_plots(plot_list)
    plt.tight_layout(pad=1.5)
    if isinstance(savefig, str):
        plt.savefig(savefig)


def mean_without_diagonal(M):
    assert M.shape[0] == M.shape[1], "M must be diagonal."
    # same as upper triangular mean
    return (M.sum() - np.diag(M).sum()) / ((M.shape[0] - 1) * (M.shape[0]))


def mean_std_upper_tri(M):
    assert M.shape[0] == M.shape[1], "M must be diagonal."
    vals = [M[i, j] for i in range(M.shape[0]) for j in range(i + 1, M.shape[1])]
    return np.mean(vals), np.std(vals)


def compute_metric_of_feature_pair(real, synt, i, j, bins=40):
    x = real[:, i]
    y = real[:, j]
    h, xe, ye, img = plt.hist2d(x=x, y=y, bins=bins)

    x = synt[:, i]
    y = synt[:, j]
    h_s, xe, ye, img = plt.hist2d(x=x, y=y, bins=[xe, ye])

    js = jensenshannon(h.flatten(), h_s.flatten())
    return js


def feature_distance_matrix(real, synt, bins=40):
    n = real.shape[1]
    M = np.zeros([n, n])
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = compute_metric_of_feature_pair(real, synt, i, j, bins=bins)
            M[j, i] = M[i, j]
    return M


def plot_corr_matrix(M, title=None, label=None, fontsize=22, skip_diag=True, ax=None):
    if ax is None:
        ax = plt
    ax.imshow(M, cmap="coolwarm", vmin=-1, vmax=1, alpha=0.75)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if i != j or not skip_diag:
                c = M[j, i]
                ax.text(
                    i,
                    j,
                    f"{c:.2f}",
                    va="center",
                    ha="center",
                    fontsize=int(fontsize * 0.4),
                )
    title_str = (
        f"{title}"
        if title is not None
        else f"Mean J-S dist. {mean_without_diagonal(M):.3f}"
    )
    title_str += f" - {label}" if label is not None else ""
    ax.set_title(title_str, fontsize=fontsize)
    ax.set_ylabel("Features")
    ax.set_xlabel("Features")
    ticks = np.arange(M.shape[0])
    ax.set_xticks(ticks, labels=ticks + 1, fontsize=6)
    ax.set_yticks(ticks, labels=ticks + 1, fontsize=6)
    return ax


def plot_all_JS_dist_matrices(
    matrices_2d, titles_2d, labels_2d, figsize=8, savefig="./plot_JS_dists.pdf"
):
    n = len(matrices_2d)
    fig, axes = plt.subplots(1, n, figsize=(n * figsize, 1 * figsize))
    for M, t, l, ax in zip(matrices_2d, titles_2d, labels_2d, axes.T):
        ax = plot_corr_matrix(M, t[0], ax=ax)
    plt.tight_layout(pad=1.5)
    if isinstance(savefig, str):
        plt.savefig(savefig)


def compute_all_JS_dist_matrices(cfg_path):
    # load config from file
    cfg_fname = pathlib.Path(cfg_path).stem
    with open(cfg_path, "r") as f:
        config_dict = yaml.safe_load(f)
    cfg = types.SimpleNamespace(**config_dict)

    datasets = ["wiki", "reddit", "mooc", "cbs"]

    matrices_2d = []
    titles_2d = []
    labels_2d = []

    for dataset in datasets:
        df_list, names_list = ld.load_test_data(dataset, cfg)
        assert names_list == [
            "real",
            "tigger",
            "dggen",
        ], "unexpected model names in names_list"

        real = df_list[0].drop(columns=["i", "j", "t"]).values
        synt = df_list[2].drop(columns=["i", "j", "t"]).values

        # compute feature histograms and JS distances
        plot_feature_hists(
            real, synt, savefig=f"./plot_feature_hists_{dataset}_{cfg_fname}.pdf"
        )

        matrices_2d += [feature_distance_matrix(real, synt)]
        titles_2d += [[dataset, dataset]]
        labels_2d += [["real", "dggen"]]

    plot_all_JS_dist_matrices(
        matrices_2d, titles_2d, labels_2d, savefig=f"./plot_JS_dists_{cfg_fname}.pdf"
    )
    return matrices_2d, titles_2d, labels_2d


def feature_relationship_matrix(fdf):
    return np.corrcoef(fdf.T)


def plot_all_correlation_matrices(
    matrices, titles, labels, figsize=8, savefig="./plot_correlations.pdf"
):
    n = len(matrices)
    fig, axes = plt.subplots(2, n, figsize=(n * figsize, 2 * figsize))
    for M, t, l, ax in zip(
        [m for mats in matrices for m in mats],
        [t for tits in titles for t in tits],
        [l for labs in labels for l in labs],
        [a for axs in axes.T for a in axs],
    ):
        plot_corr_matrix(M, t, l, ax=ax)
    plt.tight_layout(pad=1.5)
    if isinstance(savefig, str):
        plt.savefig(savefig)


def plot_correlation_AE_matrices(
    matrices, titles, labels, figsize=8, savefig="./plot_correlations_AE.pdf"
):
    n = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(n * figsize, 1 * figsize))
    for M, t, l, ax in zip(matrices, titles, labels, axes.T):
        M_r, M_s = M
        M = np.abs(M_r - M_s)
        plot_corr_matrix(M, t[0], None, ax=ax)
    plt.tight_layout(pad=1.5)
    if isinstance(savefig, str):
        plt.savefig(savefig)


def compute_all_correlation_matrices(cfg_path):
    # load config from file
    cfg_fname = pathlib.Path(cfg_path).stem
    with open(cfg_path, "r") as f:
        config_dict = yaml.safe_load(f)
    cfg = types.SimpleNamespace(**config_dict)

    datasets = ["wiki", "reddit", "mooc", "cbs"]

    matrices = []
    titles = []
    labels = []

    for dataset in datasets:
        df_list, names_list = ld.load_test_data(dataset, cfg)
        assert names_list == [
            "real",
            "tigger",
            "dggen",
        ], "unexpected model names in names_list"

        real = df_list[0].drop(columns=["i", "j", "t"]).values
        synt = df_list[2].drop(columns=["i", "j", "t"]).values
        print(f"real: {real.shape} ~~~~~~~~~~ synt: {synt.shape}")

        matrices += [
            [feature_relationship_matrix(real), feature_relationship_matrix(synt)]
        ]
        titles += [[dataset, dataset]]
        labels += [["real", "dggen"]]

    plot_all_correlation_matrices(
        matrices,
        titles,
        labels,
        figsize=8,
        savefig=f"./plot_correlations_{cfg_fname}.pdf",
    )
    plot_correlation_AE_matrices(
        matrices,
        titles,
        labels,
        figsize=8,
        savefig=f"./plot_correlations_AE_{cfg_fname}.pdf",
    )
    return matrices, titles, labels


def features_fig(cfg_path):
    # load config from file
    cfg_fname = pathlib.Path(cfg_path).stem
    with open(cfg_path, "r") as f:
        config_dict = yaml.safe_load(f)
    cfg = types.SimpleNamespace(**config_dict)

    datasets = ["wiki", "reddit", "mooc", "cbs"]

    matrices_2d = []
    titles_2d = []
    labels_2d = []

    dataset2col = {
        "wiki": 2,
        "reddit": 1,
        "mooc": 2,
        "cbs": 1,
    }

    dataset2name = {
        "cbs": "Bikeshare",
        "mooc": "MOOC",
        "reddit": "Reddit",
        "wiki": "Wikipedia",
    }

    plot_list = []
    plot_list_2d = []
    df_js = []
    for dataset in datasets:
        df_list, names_list = ld.load_test_data(dataset, cfg)
        assert names_list == [
            "real",
            "tigger",
            "dggen",
        ], "unexpected model names in names_list"

        real = df_list[0].drop(columns=["i", "j", "t"]).values
        synt = df_list[2].drop(columns=["i", "j", "t"]).values

        # compute feature histograms and JS distances
        plot_list += [
            [
                plot_feature_hist,
                (real, synt, dataset2col[dataset], dataset2name[dataset]),
            ]
        ]
        df_js += [
            [
                dataset,
                "Single feature",
                *(lambda x: [np.mean(x), np.std(x)])(
                    compute_js_dist_features(real, synt)
                ),
            ]
        ]

        M = feature_distance_matrix(real, synt)
        matrices_2d += [M]
        titles_2d += [[dataset, dataset]]
        labels_2d += [["real", "dggen"]]
        plot_list_2d += [[plot_corr_matrix, [M, None, None, 12]]]
        df_js += [[dataset, "Feature pair", *mean_std_upper_tri(M)]]

    fig, axs = arrange_plots(
        plot_list + plot_list_2d, n_cols=4, cols_scale=2.6, rows_scale=2.6
    )
    fig.savefig("./results/feature_fig.pdf")
    pd.DataFrame(df_js, columns=["Dataset", "J-S dist.", "mean", "std"]).to_csv(
        "./df_JS.csv", index=False
    )
    return matrices_2d, titles_2d, labels_2d


def create_latex_table():
    df = pd.read_csv("./df_JS.csv")
    df = df.round(3).fillna("").astype(str)
    df["JS"] = df.apply(
        lambda x: x["mean"] + (" $\pm$ " + x["std"] if x["std"] != "" else ""), axis=1
    )
    df = df.drop(columns=["mean", "std"])
    df = df.set_index(["Dataset", "J-S dist."]).unstack(level=1).droplevel(0, axis=1)
    df = df[df.columns[::-1]]
    style_str = "font-weight: bold"
    df.style.applymap_index(lambda v: "font-weight: bold;", axis=1)
    print(
        df.style.applymap_index(lambda v: "font-weight: bold;", axis=0)
        .applymap_index(lambda v: "font-weight: bold;", axis=1)
        .to_latex(convert_css=True)
    )
    return df


if __name__ == "__main__":
    # python evaluation.py eval.yaml

    features_fig(cfg_path=sys.argv[1])

    print("\n\n")
    df = create_latex_table()
