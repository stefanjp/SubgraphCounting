from subgraph_counting import pattern, conversion, datasets
from experiments.transform import (
    AnnotateSubgraphIsomorphisms,
    Select,
    NormNodeAttribute,
)
from torch_geometric.transforms.compose import Compose
import itertools
import random

def get_configs(config, count):
    keys, values = zip(*config.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return random.sample(experiments, count)

def get_dataset(dataset_id, path="./data/datasets/ZINC"):
    if dataset_id == 'ZINC-cycle-3-cycle-6':
        # load ZINC dataset with annotated subgraph isomorphisms of selected pattern(s)
        subgraph_patterns = []
        for i in range(3, 10):
            subgraph_pattern = pattern.create_cycle(i)
            subgraph_patterns.append(conversion.igraph_to_pyg_graph(subgraph_pattern))
        pre_transform = AnnotateSubgraphIsomorphisms(subgraph_patterns, True)
        transform = Compose([NormNodeAttribute("x", norm=1 / 20), Select("y", 0, [0, 3])])
        # subgraph isomorphism of cycle 6 is annotated
        return datasets.get_zinc_dataset(
            f"{path}/cycle3-9", pre_transform=pre_transform, transform=transform
        )
    elif dataset_id == 'ZINC-star-3-star-4':
        subgraph_patterns = []
        # star 3 has 4 nodes and star 9 has 10 nodes
        for i in range(4,11):
            subgraph_pattern = pattern.create_star(i)
            subgraph_patterns.append(conversion.igraph_to_pyg_graph(subgraph_pattern))

        pre_transform = AnnotateSubgraphIsomorphisms(subgraph_patterns, True)
        transform = Compose([NormNodeAttribute("x", norm=1 / 20), Select("y", 0, [0, 1])])
        return datasets.get_zinc_dataset(f'{path}/star3-9', 
            pre_transform=pre_transform,
            transform=transform
            )
    else:
        raise ValueError(f"Dataset {dataset_id} not known")
