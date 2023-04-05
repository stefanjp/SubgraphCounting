"""This module helps with graph conversion from PyTorch Geometric to igrpah."""
from typing import Callable
from torch_geometric.data import Data
import torch
import numpy as np
import igraph
from subgraph_counting.graph import Graph

def graph_to_igraph(graph: Graph):
    return igraph.Graph(n=graph.get_num_nodes(), edges=graph.get_edge_indices(), directed=graph.is_directed())

def igraph_to_pyg_graph(igraph: igraph.Graph):
    edges = []
    for edge in igraph.es:
        edges.append(torch.as_tensor(edge.tuple))
    edge_index = torch.stack(edges, dim=1)
    data = Data(edge_index=edge_index)
    data.num_nodes = igraph.vcount()
    return data

def _tensor_to_color(tensor: torch.Tensor):
    array = tensor.numpy()
    if array.size == 1 and np.issubdtype(array.dtype, np.integer):
        return array.item()
    else:
        return hash(array.data.tobytes())
    
def _tensor_to_label(tensor: torch.Tensor):
    array = tensor.numpy()
    if array.size == 1:
        return str(array.item())
    else:
        return str(array)

def to_color(tensor: torch.Tensor,
  conversion_function: Callable[[torch.Tensor], int] | None = None) -> list[int]:
    if conversion_function is None:
        conversion_function = _tensor_to_color
    return [conversion_function(data) for data in tensor]

def to_label(tensor: torch.Tensor,
  conversion_function: Callable[[torch.Tensor], str] | None = None) -> list[str]:
    if conversion_function is None:
        conversion_function = _tensor_to_label
    return [conversion_function(data) for data in tensor]
    
def get_duplicate_edge_mask(edge_index: torch.Tensor,
  order_strict:bool = False):
    """Get boolean torch mask. Mask is of shape [N],
        where N is the number of edges. Edges that appear more 
        than once, between two nodes, are marked as True in the mask.
        Pass the inverted mask to conversion functions,
        if you want to delete duplicate edges.

    Parameters
    ----------
    edge_index : torch.Tensor
        edge_index of PyG graph of shape [2, edge_count].
    order_strict:bool = False
        If order order_strict == True, edges are considered as duplicates
        only if there exists more than one edge having the same source and
        target node. Otherwise marking duplicates is order independent.
    """
    edge_count = edge_index.shape[1]
    mask = torch.zeros(edge_count, dtype=torch.bool)
    edges = set()
    for i in range(edge_count):
        source = edge_index[0, i].item()
        target = edge_index[1, i].item()
        if order_strict:
            edge = (source, target)
        else:
            edge = (min(source, target), max(source, target))
        if edge in edges:
            mask[i] = torch.as_tensor([1])
        else:
            edges.add(edge)
    return mask
