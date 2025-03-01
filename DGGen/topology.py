"""
Adapted from Gupta et al. 2022: https://github.com/data-iitd/tigger
"""

from metrics.metric_utils import get_numpy_matrix_from_adjacency
from metrics.metrics import (
    compute_graph_statistics,
)
import pickle
import numpy as np
import pandas as pd
import warnings
import math
import sys


def main(dataset):

    def get_edges_from_adj_graph(graph):
        s = set()
        for start, adj_list in graph.items():
            for end, value in adj_list.items():
                if value > 0:
                    start, end = min(start, end), max(start, end)
                s.add("_".join([str(start), str(end)]))
        return s

    original_graphs_path = (
        f"results/synthetic_data/tigger/source_graph_pickles/{dataset}.pkl"
    )
    sampled_graphs_path = (
        f"results/synthetic_data/baseline_compatible/snapshots/{dataset}.pkl"
    )

    original_graphs = pickle.load(open(original_graphs_path, "rb"))
    sampled_graphs = pickle.load(open(sampled_graphs_path, "rb"))

    print("Length of original and sampled,", len(original_graphs), len(sampled_graphs))
    commons = []
    for i in range(0, min(len(sampled_graphs), len(original_graphs))):
        sgraph = sampled_graphs[i]
        ograph = original_graphs[i]
        sgraphedges = get_edges_from_adj_graph(sgraph)
        ographedges = get_edges_from_adj_graph(ograph)
        len_o = len(ographedges)
        len_common = len(ographedges.intersection(sgraphedges))
        if len_o != 0:
            commons.append(len_common * 100.0 / len_o)

    mean_edge_intersection = np.mean(commons)
    median_edge_intersection = np.median(commons)

    result = {}
    result["edge_diversity"] = np.median(commons)

    df_metric = []
    df_val = []

    df_metric.append("mean_edge_overlap")
    df_val.append(mean_edge_intersection)
    df_metric.append("median_edge_overlap")
    df_val.append(median_edge_intersection)

    labels = []
    old_stats = []
    new_stats = []
    ct = 0
    labels = []

    warnings.filterwarnings("ignore")
    for ct in range(min(len(sampled_graphs), len(original_graphs))):
        original_matrix, _, _ = get_numpy_matrix_from_adjacency(original_graphs[ct])
        sampled_matrix, _, _ = get_numpy_matrix_from_adjacency(sampled_graphs[ct])

        assert (original_matrix == original_matrix.T).all()

        if original_matrix.shape[0] > 1 and sampled_matrix.shape[0] > 1:
            labels.append(ct)
            old_graph_stats = compute_graph_statistics(original_matrix)
            new_graph_stats = compute_graph_statistics(sampled_matrix)
            old_stats.append(old_graph_stats)
            new_stats.append(new_graph_stats)

    actual_graph_result = {}
    for metric in old_stats[0].keys():
        actual_graph_metrics = [item[metric] for item in old_stats]
        sampled_graph_metrics = [item[metric] for item in new_stats]
        abs_error = [
            abs(a - b) * 1.00
            for a, b in zip(actual_graph_metrics, sampled_graph_metrics)
        ]
        infs = [item for item in abs_error if (pd.isnull(item) or math.isinf(item))]
        if len(infs) > 0:
            print("infs found, ", len(infs), metric)
        abs_error = [
            item for item in abs_error if (not pd.isnull(item) and not math.isinf(item))
        ]
        actual_graph_metrics = [
            item
            for item in actual_graph_metrics
            if (not pd.isnull(item) and not math.isinf(item))
        ]
        print("Actual graph metrices", len(actual_graph_metrics))
        result["{}".format(metric)] = np.median(abs_error)
        actual_graph_result["{}".format(metric)] = np.median(actual_graph_metrics)

    print(result)
    nums = []
    for metric in [
        "edge_diversity",
        "d_mean",
        "wedge_count",
        "triangle_count",
        "power_law_exp",
        "rel_edge_distr_entropy",
        "LCC",
        "n_components",
        "clustering_coefficient",
        "betweenness_centrality_mean",
        "closeness_centrality_mean",
    ]:
        nums.append(result[metric])

    results = [np.round(item, 16) for item in nums]
    print("median values:")
    nums = "& ".join(["$" + str(item) + "$" for item in results])
    print(nums)


if __name__ == "__main__":
    assert sys.argv[-1] in [
        "reddit",
        "wikipedia",
        "mooc",
        "lastfm",
        "bikeshare",
    ], "Choose a valid dataset."

    dataset = sys.argv[-1]
    main(dataset)
