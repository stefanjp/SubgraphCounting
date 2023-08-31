"""Graph representation and conversion"""

from __future__ import annotations  # enable typehint to incomplete class
from torch_geometric.data import Data
from subgraph_counting import conversion, plot
import torch_geometric.utils.undirected as undirected
from typing import Callable, Tuple
import igraph
import torch


class Graph:
    """Graph representation tailored to subgraph counting and graph plotting"""

    def __init__(
        self,
        pyg_graph: Data,
        pyg_to_undirected: bool = False,
        pyg_to_undirected_reduce: str = "mean",
        node_attribute: str | None = None,
        node_attribute_to_color: Callable[[torch.Tensor], int] | None = None,
        node_attribute_to_label: Callable[[torch.Tensor], str] | None = None,
        edge_attribute: str | None = None,
        edge_attribute_to_color: Callable[[torch.Tensor], int] | None = None,
        edge_attribute_to_label: Callable[[torch.Tensor], str] | None = None,
    ):
        if node_attribute is not None:
            assert hasattr(pyg_graph, node_attribute)
        if edge_attribute is not None:
            assert hasattr(pyg_graph, edge_attribute)

        self._graph_is_directed = pyg_graph.is_directed()
        if pyg_to_undirected:
            self._graph_is_directed = False
            if edge_attribute is not None:
                (edge_index, edge_attributes) = undirected.to_undirected(
                    pyg_graph.edge_index,
                    pyg_graph[edge_attribute],
                    num_nodes=pyg_graph.num_nodes,
                    reduce=pyg_to_undirected_reduce,
                )
                duplicate_edge_mask = conversion.get_duplicate_edge_mask(edge_index)
                mask = ~duplicate_edge_mask
                self._edge_index = edge_index[:, mask]
                self._edge_attributes = edge_attributes[mask, ...]
            else:
                edge_index = undirected.to_undirected(
                    pyg_graph.edge_index,
                    num_nodes=pyg_graph.num_nodes,
                    reduce=pyg_to_undirected_reduce,
                )
                duplicate_edge_mask = conversion.get_duplicate_edge_mask(edge_index)
                mask = ~duplicate_edge_mask
                self._edge_index = edge_index[:, mask]
                self._edge_attributes = None

        else:
            self._edge_index = pyg_graph.edge_index
            self._edge_attributes = (
                pyg_graph[edge_attribute] if edge_attribute is not None else None
            )

        self._num_nodes = (
            pyg_graph.num_nodes
        )  # keep all nodes, even if edges are removed
        self._node_attributes = (
            pyg_graph[node_attribute] if node_attribute is not None else None
        )
        self._node_attribute_to_color = node_attribute_to_color
        self._node_attribute_to_label = node_attribute_to_label
        self._edge_attribute_to_color = edge_attribute_to_color
        self._edge_attribute_to_label = edge_attribute_to_label
        self._edge_index_list_of_tuples = None
        self._node_colors = None
        self._node_labels = None
        self._edge_colors = None
        self._edge_labels = None

    def get_node_colors(self) -> list[int]:
        if self._node_attributes is not None and self._node_colors is None:
            self._node_colors = conversion.to_color(
                self._node_attributes, self._node_attribute_to_color
            )

        return self._node_colors

    def get_node_labels(self) -> list[str]:
        if self._node_attributes is not None and self._node_labels is None:
            self._node_labels = conversion.to_label(
                self._node_attributes, self._node_attribute_to_label
            )

        return self._node_labels

    def get_edge_colors(self) -> list[int]:
        if self._edge_attributes is not None and self._edge_colors is None:
            self._edge_colors = conversion.to_color(
                self._edge_attributes, self._edge_attribute_to_color
            )

        return self._edge_colors

    def get_edge_labels(self) -> list[str]:
        if self._edge_attributes is not None and self._edge_labels is None:
            self._edge_labels = conversion.to_label(
                self._edge_attributes, self._edge_attribute_to_label
            )

        return self._edge_labels

    def get_edge_indices(self) -> list[Tuple[int, int]]:
        if self._edge_index_list_of_tuples is None:
            if self._edge_index is not None:
                edges = torch.split(self._edge_index, 1, dim=1)
                self._edge_index_list_of_tuples = [
                    tuple(edge[:, 0].tolist()) for edge in edges
                ]
            else:
                self._edge_index_list_of_tuples = []
        return self._edge_index_list_of_tuples

    def get_num_nodes(self) -> int:
        return self._num_nodes
    
    def get_num_edges(self) -> int:
        return len(self.get_edge_indices())

    def count_subisomorphisms(self, subgraph: Graph):
        graph_ig = igraph.Graph(n=self.get_num_nodes(), edges=self.get_edge_indices())
        subgraph_ig = igraph.Graph(
            n=subgraph.get_num_nodes(), edges=subgraph.get_edge_indices()
        )
        subisomorphism_count = graph_ig.count_subisomorphisms_vf2(
            subgraph_ig,
            color1=self.get_node_colors(),
            color2=subgraph.get_node_colors(),
            edge_color1=self.get_edge_colors(),
            edge_color2=subgraph.get_edge_colors(),
        )
        automorphisms = subgraph_ig.count_automorphisms_vf2(
            color=subgraph.get_node_colors(), edge_color=subgraph.get_edge_colors()
        )
        return subisomorphism_count / automorphisms

    def is_directed(self):
        """Returns True, if the graph is a directed graph.
        """
        return self._graph_is_directed

    def get_subisomorphisms(self, subgraph: Graph) -> list:
        graph_ig = igraph.Graph(
            n=self.get_num_nodes(),
            edges=self.get_edge_indices(),
            directed=self.is_directed(),
        )
        subgraph_ig = igraph.Graph(
            n=subgraph.get_num_nodes(),
            edges=subgraph.get_edge_indices(),
            directed=subgraph.is_directed(),
        )
        subisomorphisms = graph_ig.get_subisomorphisms_vf2(
            subgraph_ig,
            color1=self.get_node_colors(),
            color2=subgraph.get_node_colors(),
            edge_color1=self.get_edge_colors(),
            edge_color2=subgraph.get_edge_colors(),
        )
        return subisomorphisms

    def plot(
        self,
        fname: str | None = None,
        node_colors: bool = False,
        node_labels: bool = False,
        edge_colors: bool = False,
        edge_labels: bool = False,
        subgraph: Graph | None = None,
        visual_style: dict | None = None,
    ):
        """Plot graph with optional node/edge colors and labels"""
        if not visual_style:
            visual_style = {}
        if node_colors:
            visual_style["vertex_color"] = plot.attributes_to_hexcolor(
                self.get_node_colors()
            )
        else:
            if "vertex_color" not in visual_style:
                visual_style["vertex_color"] ='white'
        if node_labels:
            visual_style["vertex_label"] = self.get_node_labels()
        if edge_colors:
            visual_style["edge_color"] = plot.attributes_to_hexcolor(
                self.get_edge_colors()
            )
        if edge_labels:
            visual_style["edge_label"] = self.get_edge_labels()
        if subgraph is not None:
            subisomorphisms = self.get_subisomorphisms(subgraph)
            sub_nodes, sub_edges = _get_subgraph_mask(self, subgraph, subisomorphisms)
            visual_style["vertex_color"] = [
                "red" if mask else "white" for mask in sub_nodes
            ]
            visual_style["edge_width"] = [3 if mask else 1 for mask in sub_edges]
        return plot.plot_graph(_graph_to_igraph(self), fname, visual_style)

def _graph_to_igraph(graph: Graph):
    return igraph.Graph(
        n=graph.get_num_nodes(),
        edges=graph.get_edge_indices(),
        directed=graph.is_directed(),
    )

def _get_subgraph_mask(graph: Graph, subgraph: Graph, subgraph_node_lists: list):
    """Get a mask of all nodes and edges of the subgraph in graph, using the subgraph_node_list generated by igraph.

    Parameters
    ----------
    graph : Graph
    subgraph : Graph
    subgraph_node_lists : list

    Returns
    -------
    list, list
        Returns node_mask, edge_mask. Two lists of boolean 
        values where True values mark nodes and edges of the subgraph.
    """
    subgraph_edges = set()
    subgraph_nodes = set()
    directed = graph.is_directed()
    for subgraph_matching in subgraph_node_lists:
        subgraph_nodes = subgraph_nodes.union(subgraph_matching)
        g_sub_to_g_vertex = {i: v for i, v in enumerate(subgraph_matching)}
        if directed:
            for sub_edge in subgraph.get_edge_indices():
                subgraph_edges.add(
                    (g_sub_to_g_vertex[sub_edge[0]], g_sub_to_g_vertex[sub_edge[1]])
                )
            edge_mask = [
                True if edge in subgraph_edges else False
                for edge in graph.get_edge_indices()
            ]
        else:
            for sub_edge in subgraph.get_edge_indices():
                source = g_sub_to_g_vertex[sub_edge[0]]
                target = g_sub_to_g_vertex[sub_edge[1]]
                subgraph_edges.add((min(source, target), max(source, target)))
            edge_mask = [
                True if tuple(sorted(edge)) in subgraph_edges else False
                for edge in graph.get_edge_indices()
            ]
    node_mask = [
        True if node in subgraph_nodes else False
        for node in range(graph.get_num_nodes())
    ]
    return node_mask, edge_mask
