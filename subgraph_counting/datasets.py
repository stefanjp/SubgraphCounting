"""Convenience functions for graph generation used mainly for testing"""
import torch
from torch_geometric.datasets import ZINC
from torch_geometric.data import Data
from torch_geometric.utils.random import erdos_renyi_graph

def get_zinc_dataset(root:str="./data/datasets/ZINC") -> dict[str, ZINC]:
    """Load and store ZINC dataset

    Parameters
    ----------
    root : str, optional
        root directory for file storage, by default "./data/datasets/ZINC"

    Returns
    -------
    dict[str, ZINC]
        returns dict of train, val and test data.
        keys are "train", "val", "test". values are the datasets
    """
    return {split: ZINC(root=root, subset=True, split=split) for split in ["train", "val", "test"]}

def get_zachary_graph() -> Data:
    """Graph from http://www1.ind.ku.dk/complexLearning/zachary1977.pdf paper"""
    row = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4,
        5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 10, 10,
        10, 11, 12, 12, 13, 13, 13, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17,
        18, 18, 19, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 23, 23, 23, 24,
        24, 24, 25, 25, 25, 26, 26, 27, 27, 27, 27, 28, 28, 28, 29, 29, 29,
        29, 30, 30, 30, 30, 31, 31, 31, 31, 31, 31, 32, 32, 32, 32, 32, 32,
        32, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33,
        33, 33, 33, 33, 33, 33
    ]
    col = [
        1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 17, 19, 21, 31, 0, 2, 3, 7,
        13, 17, 19, 21, 30, 0, 1, 3, 7, 8, 9, 13, 27, 28, 32, 0, 1, 2, 7,
        12, 13, 0, 6, 10, 0, 6, 10, 16, 0, 4, 5, 16, 0, 1, 2, 3, 0, 2, 30,
        32, 33, 2, 33, 0, 4, 5, 0, 0, 3, 0, 1, 2, 3, 33, 32, 33, 32, 33, 5,
        6, 0, 1, 32, 33, 0, 1, 33, 32, 33, 0, 1, 32, 33, 25, 27, 29, 32,
        33, 25, 27, 31, 23, 24, 31, 29, 33, 2, 23, 24, 33, 2, 31, 33, 23,
        26, 32, 33, 1, 8, 32, 33, 0, 24, 25, 28, 32, 33, 2, 8, 14, 15, 18,
        20, 22, 23, 29, 30, 31, 33, 8, 9, 13, 14, 15, 18, 19, 20, 22, 23,
        26, 27, 28, 29, 30, 31, 32
    ]
    edge_index = torch.as_tensor([row, col])
    x = torch.arange(1,35)
    edge_attr = torch.arange(1, len(col) + 1)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

def get_random_graph(num_nodes: int, edge_prob: float, directed: bool = False):
    """Get random PyG graph"""
    graph_pyg = Data(edge_index=erdos_renyi_graph(num_nodes, edge_prob, directed))
    graph_pyg.num_nodes = num_nodes
    return graph_pyg