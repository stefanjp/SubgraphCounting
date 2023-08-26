from subgraph_counting import pattern, conversion, datasets, graph
from torch_geometric.data import Data
import torch
import matplotlib.pyplot as plt

PLOT_ROOT = "data/figures"
ZINC_DATASET = "data/datasets/ZINC/original"


def count_cycles_in_zinc_dataset():
    """Execute code that counts cycles of size 8 in all graphs of ZINC dataset.
    The ZINC dataset is interpreted as undirected graphs without vertex or edge attributes.
    """
    # get ZINC dataset
    dataset_pyg = datasets.get_zinc_dataset(ZINC_DATASET)["train"]
    pattern_size = 10
    subgraph_pattern = conversion.igraph_to_pyg_graph(
        pattern.create_cycle(pattern_size)
    )
    pattern_counts = {}
    for i, graph_pyg in enumerate(dataset_pyg):
        g = graph.Graph(graph_pyg, True)
        pattern_counts[i] = g.count_subisomorphisms(graph.Graph(subgraph_pattern))
        if i % 200 == 0:
            print(f"Progress: {i / len(dataset_pyg) * 100:.1f}%", end="\r")


def count_cycles_with_attributes_in_zinc_dataset():
    """Execute code that counts subgraph structure with attributres"""
    # get ZINC dataset
    dataset_pyg = datasets.get_zinc_dataset(ZINC_DATASET)["train"]
    subgraph_pattern = conversion.igraph_to_pyg_graph(pattern.create_cycle(6))
    subgraph_pattern = Data(
        x=torch.as_tensor([0 for _ in range(6)]),
        edge_index=subgraph_pattern.edge_index,
        edge_attr=torch.as_tensor([1, 2, 1, 2, 1, 2]),
    )
    for i, graph_pyg in enumerate(dataset_pyg):
        g = graph.Graph(graph_pyg, True, node_attribute="x", edge_attribute="edge_attr")
        fig, _ = g.plot(f"{PLOT_ROOT}/zink_train_{i}_attributes.png")
        subgraph = graph.Graph(
            subgraph_pattern, True, node_attribute="x", edge_attribute="edge_attr"
        )
        subisomorphisms = g.get_subisomorphisms(subgraph)
        plt.close(fig)
        if subisomorphisms:
            fig, _ = g.plot(
                f"{PLOT_ROOT}/zink_train_{i}_cycle_8.png",
                node_colors=True,
                node_labels=True,
                edge_labels=True,
                subgraph=subgraph,
            )
            plt.close(fig)



if __name__ == "__main__":
    # execute graph substructure counting examples
    count_cycles_in_zinc_dataset()
    count_cycles_with_attributes_in_zinc_dataset()

