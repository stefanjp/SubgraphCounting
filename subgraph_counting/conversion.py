"""This module helps with graph conversion from PyTorch Geometric to igrpah."""
from typing import Callable, Tuple
from torch_geometric.data import Data
import torch
import numpy as np
from igraph import Graph

def pyg_to_igraph(pyg_graph: Data, edge_mask: torch.Tensor | None = None, to_undirected: bool = False) -> Graph:
    """Convert PyTorch Geometric graph to igraph. Does not convert vertex or node attributes.
    
    Parameters
    ----------
    pyg_graph : torch_geometric.data.Data
        PyG graph.
    edge_mask: torch.Tensor | None = None
        boolean tensor which contains True for all edges that should remain in graph,
        and False for all edges that will be deleted!
    to_undirected: bool = False
        Creates undirected graph. Does not add/remove edges!
    
    Returns
    ----------
    Graph
        igraph object.
    """
    num_nodes = pyg_graph.num_nodes
    if edge_mask is not None:
        edge_index = pyg_graph.edge_index[:, edge_mask]
    else:
        edge_index = pyg_graph.edge_index
    
    edges = torch.split(edge_index, 1, dim=1)
    edges = [tuple(edge[:, 0].tolist()) for edge in edges]
    return Graph(n=num_nodes, edges=edges, directed=pyg_graph.is_directed() and not to_undirected)

def _tensor_to_color(array: np.ndarray):
    if array.shape[0] == 1:
        return array.item()
    else:
        return hash(array.data.tobytes())
    
def _tensor_to_label(array: np.ndarray):
    if array.shape[0] == 1:
        return str(array.item())
    else:
        return str(array)

def _pyg_attributes(tensor: torch.Tensor,
  to_color: Callable[[torch.Tensor], int] | None = None,
  to_label: Callable[[torch.Tensor], str] | None = None,
  attr_mask: torch.Tensor | None = None) -> Tuple[list[int], list[str]]:
    if to_color is None:
        to_color = _tensor_to_color
    if to_label is None:
        to_label = _tensor_to_label
    
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(1)

    if attr_mask is not None:
        tensor = tensor[attr_mask, :]

    array = tensor.numpy()
    node_colors = [to_color(data) for data in array]
    node_labels = [to_label(data) for data in array]

    return (node_colors, node_labels)

def pyg_get_node_attributes(pyg_graph: Data,
  attr_key: str = 'x',
  to_color: Callable[[torch.Tensor], int] | None = None,
  to_label: Callable[[torch.Tensor], str] | None = None,
  ) -> Tuple[list[int], list[str]]:
    """Get node colors and labels from PyG Data.
    
    Parameters
    ----------
    pyg_graph : torch_geometric.data.Data
        PyG graph.
    attr_key: str = 'x'
        Key to node attributes
    to_color: Callable[[torch.Tensor], int] | None = None
        Function that converts a node attribute to a node color
    to_label: Callable[[torch.Tensor], str] | None = None
        Function that converts a node attribute to a node label

    Returns
    ----------
    Tuple[list[int], list[str]]
        list of colors and labels for each node.
    """
    return _pyg_attributes(pyg_graph[attr_key], to_color, to_label)

def pyg_get_edge_attributes(pyg_graph: Data,
  attr_key: str = 'edge_attr',
  to_color: Callable[[torch.Tensor], int] | None = None,
  to_label: Callable[[torch.Tensor], str] | None = None,
  attr_mask: torch.Tensor | None = None,
  ) -> Tuple[list[int], list[str]]:
    """Get edge colors and labels from PyG Data.
    
    Parameters
    ----------
    pyg_graph : torch_geometric.data.Data
        PyG graph.
    attr_key: str = 'edge_attr'
        Key to node attributes
    to_color: Callable[[torch.Tensor], int] | None = None
        Function that converts an edge attribute to an edge color
    to_label: Callable[[torch.Tensor], str] | None = None
        Function that converts an edge attribute to an edge label
    attr_mask: torch.Tensor | None = None
        Boolean torch filter mask. Filters edge attributes. Mask is of shape [N],
        where N is the number of edges. All values that contain False are filtered.
        attr_mask has the same order as edge_index.
    
    Returns
    ----------
    Tuple[list[int], list[str]]
        list of colors and labels for each edge.
    """
    return _pyg_attributes(pyg_graph[attr_key], to_color, to_label, attr_mask)

def pyg_get_duplicate_edge_mask(pyg_graph: Data,
  order_strict:bool = False):
    """Get boolean torch mask. Mask is of shape [N],
        where N is the number of edges. Edges that appear more 
        than once, between two nodes, are marked as True in the mask.
        Pass the inverted mask to conversion functions,
        if you want to delete duplicate edges.

    Parameters
    ----------
    pyg_graph : torch_geometric.data.Data
        PyG graph.
    order_strict:bool = False
        If order order_strict == True, edges are considered as duplicates
        only if there exists more than one edge having the same source and
        target node. Otherwise marking duplicates is order independent.
    """
    edge_count = pyg_graph.edge_index.shape[1]
    mask = torch.zeros(edge_count, dtype=torch.bool)
    edges = set()
    for edge_index in range(edge_count):
        source = pyg_graph.edge_index[0, edge_index].item()
        target = pyg_graph.edge_index[1, edge_index].item()
        if order_strict:
            edge = (source, target)
        else:
            edge = (min(source, target), max(source, target))
        if edge in edges:
            mask[edge_index] = torch.as_tensor([1])
        else:
            edges.add(edge)
    return mask
    
