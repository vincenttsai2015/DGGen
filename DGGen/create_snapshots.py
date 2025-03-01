"""
Adapted from Gupta et al. 2022: https://github.com/data-iitd/tigger
"""

from metrics.metric_utils import (
    get_numpy_matrix_from_adjacency,
    get_total_nodes_and_edges_from_temporal_adj_list_in_time_range,
    get_adj_origina_graph_from_original_temporal_graph,
)
from metrics.tgg_utils import *
import pandas as pd
import pickle
from collections import defaultdict
import sys


def main(dataset):
    data_path = f"results/synthetic_data/baseline_compatible/{dataset}.csv"
    out_dir = "results/synthetic_data/baseline_compatible/snapshots/"
    time_window = 500
    fname = dataset

    undirected = True

    data = pd.read_csv(data_path)

    data = data[["start", "end", "days"]]
    node_set = set(data["start"]).union(set(data["end"]))
    node_set.update("end_node")
    max_days = max(data["days"])
    data = data.sort_values(by="days", inplace=False)

    temporal_graph_original = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: 0))
    )
    for start, end, day in data[["start", "end", "days"]].values:
        temporal_graph_original[day][start][end] += 1
        if undirected:
            temporal_graph_original[day][end][start] += 1

    target_node_counts = []
    target_edge_counts = []
    time_labels = []
    for start_time in range(1, max_days, time_window):
        tp, node_count = get_total_nodes_and_edges_from_temporal_adj_list_in_time_range(
            temporal_graph_original, start_time, start_time + time_window - 1
        )
        if undirected:
            tp = int(tp / 2)

        target_edge_counts.append(tp)
        target_node_counts.append(node_count)

    original_graphs = []
    for start_time in range(1, max_days, time_window):
        time_labels.append(start_time)
        original_graphs.append(
            get_adj_origina_graph_from_original_temporal_graph(
                temporal_graph_original, start_time, start_time + time_window - 1
            )
        )
    degree_distributions = []
    for i, graph in enumerate(original_graphs):
        temp, _, _ = get_numpy_matrix_from_adjacency(graph)
        degree_distributions.append(list(temp.sum(axis=0)))

    pickle.dump(original_graphs, open(out_dir + f"/{fname}.pkl", "wb"))


if __name__ == "__main__":
    dataset = sys.argv[1]
    main(dataset)
