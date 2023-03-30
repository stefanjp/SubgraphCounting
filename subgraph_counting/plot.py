"""Plotting functionality tailored for graph and subgraph visualization"""
from typing import Any
from igraph import Graph
import igraph as ig
import matplotlib.pyplot as plt
import matplotlib

def _accessible_colors_95():
    """Cycle Sasha Trubetskoy's 20 simple colors in your Matplotlib plots
    https://sashamaps.net/docs/resources/20-colors/
    """
    return [
        '#e6194B', '#3cb44b', '#ffe119',
        '#4363d8', '#f58231', '#911eb4',
        '#42d4f4', '#f032e6', '#bfef45',
        '#fabed4', '#469990', '#dcbeff',
        '#9A6324', '#fffac8', '#800000',
        '#aaffc3', '#808000', '#ffd8b1',
        '#000075', '#a9a9a9', '#ffffff', '#000000']

def attributes_to_hexcolor(attributes: list[int]):
    simple_colors = _accessible_colors_95()
    unique_attributes = list(set(attributes))
    if len(unique_attributes) <= len(simple_colors):
        attr_map = {unique_attributes[i]: simple_colors[i] for i in range(len(unique_attributes))}
    else:
        cmap = plt.cm.get_cmap('viridis')
        attr_map = {unique_attributes[i] : matplotlib.colors.to_hex(cmap(i)) for i in range(len(unique_attributes))}
    return [attr_map[attr] for attr in attributes]

def _plot_graph(
  graph: Graph,
  fname: str | None = None,
  visual_style: dict[str, Any] | None = None):
    """plot graph with style attributes"""
    fig, axes = plt.subplots()
    if visual_style:
        ig.plot(graph, **visual_style, target=axes)
    else:
        ig.plot(graph, target=axes)
    if fname:
        fig.savefig(fname, facecolor='white')
    return fig, axes

def plot_graph(
  graph: Graph,
  fname: str | None = None,
  node_attributes: list[int] | None = None,
  node_labels: list[str] | None = None,
  edge_attributes: list[int] | None = None,
  edge_labels: list[str] | None = None,
  subgraph_node_mask: list | None = None,
  subgraph_edge_mask: list | None = None,
  visual_style: dict | None = None):
    """Plot graph with optional node/edge colors and labels"""
    assert(not (subgraph_node_mask and node_attributes))
    if not visual_style:
        visual_style = {}
    if node_attributes:
        visual_style["vertex_color"] = attributes_to_hexcolor(node_attributes)
    if node_labels:
        visual_style["vertex_label"] = node_labels
    if edge_attributes:
        visual_style["edge_color"] = attributes_to_hexcolor(edge_attributes)
    if edge_labels:
        visual_style["edge_label"] = edge_labels
    if subgraph_node_mask:
        visual_style["vertex_color"] = ['red' if mask else 'white' for mask in subgraph_node_mask]
    if subgraph_edge_mask:
        visual_style["edge_width"] = [3 if mask else 1 for mask in subgraph_edge_mask]
    return _plot_graph(graph, fname, visual_style)
