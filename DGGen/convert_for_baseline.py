import sys
import os
import glob
import torch
import pandas as pd


def main(in_file, out_file):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    class Graph:
        def __init__(self, src, dst, t):
            self.src = src.tolist()
            self.dst = dst.tolist()
            self.t = t.tolist()

    graph_ = torch.load(f"{in_file}", map_location=device)
    graph = Graph(graph_.src, graph_.dst, graph_.t)

    # convert to 1-index
    graph.src = [x + 1 for x in graph.src]
    graph.dst = [x + 1 for x in graph.dst]
    graph.t = [x + 1 for x in graph.t]

    data = {"start": graph.src, "end": graph.dst, "days": graph.t}
    df = pd.DataFrame(data)

    df.to_csv(f"{out_file}", index=True)


if __name__ == "__main__":

    # it is not necessary to change path hardcodes if using instructions from README
    dataset = sys.argv[1]
    search_dir = "./results/synthetic_data/"

    matching_files = glob.glob(os.path.join(search_dir, f"{dataset}*.pt"))

    if len(matching_files) == 1:
        IN_FILE = matching_files[0]
    else:
        raise FileNotFoundError(
            f"Expected exactly one file starting with '{dataset}' and ending with '.pt', but found {len(matching_files)}."
        )

    OUT_FILE = f"./results/synthetic_data/baseline_compatible/{dataset}.csv"

    main(IN_FILE, OUT_FILE)
