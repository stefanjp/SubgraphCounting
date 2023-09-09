from subgraph_counting import pattern, conversion, datasets
from experiments.transform import (
    AnnotateSubgraphIsomorphisms,
    Select,
    NormNodeAttribute,
)
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import FakeDataset
from torch_geometric.transforms.compose import Compose
import itertools
import random


def get_configs(config, count=None):
    keys, values = zip(*config.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    if count is None:
        return experiments
    else:
        return random.sample(experiments, count)


def get_dataset(dataset_id, path="./data/datasets/ZINC"):
    if dataset_id == "ZINC-cycle-3-cycle-6":
        # load ZINC dataset with annotated subgraph isomorphisms of selected pattern(s)
        subgraph_patterns = []
        for i in range(3, 10):
            subgraph_pattern = pattern.create_cycle(i)
            subgraph_patterns.append(conversion.igraph_to_pyg_graph(subgraph_pattern))
        pre_transform = AnnotateSubgraphIsomorphisms(subgraph_patterns, True)
        transform = Compose(
            [NormNodeAttribute("x", norm=1 / 20), Select("y", 0, [0, 3])]
        )
        # subgraph isomorphism of cycle 6 is annotated
        return datasets.get_zinc_dataset(
            f"{path}/cycle3-9", pre_transform=pre_transform, transform=transform
        )
    elif (
        dataset_id == "ZINC-cycle-6-node-features"
    ):  # takes node features into account for graph substructure counting
        # load ZINC dataset with annotated subgraph isomorphisms of selected pattern(s)
        subgraph_pattern = conversion.igraph_to_pyg_graph(pattern.create_cycle(6))
        subgraph_pattern = Data(
            x=torch.as_tensor([0 for _ in range(6)]),
            edge_index=subgraph_pattern.edge_index,
        )
        pre_transform = AnnotateSubgraphIsomorphisms(
            [subgraph_pattern], True, node_attribute="x"
        )
        transform = Compose([NormNodeAttribute("x", norm=1 / 20)])
        # subgraph isomorphism of cycle 6 is annotated
        return datasets.get_zinc_dataset(
            f"{path}/cycle6", pre_transform=pre_transform, transform=transform
        )
    elif dataset_id == "ZINC-cycle-3":
        # load ZINC dataset with annotated subgraph isomorphisms of selected pattern(s)
        subgraph_patterns = []
        for i in range(3, 10):
            subgraph_pattern = pattern.create_cycle(i)
            subgraph_patterns.append(conversion.igraph_to_pyg_graph(subgraph_pattern))
        pre_transform = AnnotateSubgraphIsomorphisms(subgraph_patterns, True)
        transform = Compose([NormNodeAttribute("x", norm=1 / 20), Select("y", 0, [0])])
        # subgraph isomorphism of cycle 6 is annotated
        return datasets.get_zinc_dataset(
            f"{path}/cycle3-9", pre_transform=pre_transform, transform=transform
        )
    elif dataset_id == "ZINC-star-3-star-4":
        subgraph_patterns = []
        # star 3 has 4 nodes and star 9 has 10 nodes
        for i in range(4, 11):
            subgraph_pattern = pattern.create_star(i)
            subgraph_patterns.append(conversion.igraph_to_pyg_graph(subgraph_pattern))

        pre_transform = AnnotateSubgraphIsomorphisms(subgraph_patterns, True)
        transform = Compose(
            [NormNodeAttribute("x", norm=1 / 20), Select("y", 0, [0, 1])]
        )
        return datasets.get_zinc_dataset(
            f"{path}/star3-9", pre_transform=pre_transform, transform=transform
        )
    elif dataset_id == "random-cycle-3-cycle-6":
        subgraph_patterns = []
        # star 3 has 4 nodes and star 9 has 10 nodes
        for i in [3, 6]:
            subgraph_pattern = pattern.create_cycle(i)
            subgraph_patterns.append(conversion.igraph_to_pyg_graph(subgraph_pattern))
        pre_transform = AnnotateSubgraphIsomorphisms(subgraph_patterns, True)
        return {
            "train": FakeDataset(
                1000,
                avg_num_nodes=20,
                avg_degree=3,
                num_channels=1,
                num_classes=1,
                is_undirected=False,
                transform=pre_transform,
            ),
            "val": FakeDataset(
                100,
                avg_num_nodes=20,
                avg_degree=3,
                num_channels=1,
                num_classes=1,
                is_undirected=False,
                transform=pre_transform,
            ),
            "test": FakeDataset(
                100,
                avg_num_nodes=20,
                avg_degree=3,
                num_channels=1,
                num_classes=1,
                is_undirected=False,
                transform=pre_transform,
            ),
        }
    else:
        raise ValueError(f"Dataset {dataset_id} not known")
