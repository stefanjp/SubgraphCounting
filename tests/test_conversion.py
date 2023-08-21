"""Test graph and attribute conversion"""
import unittest
import torch
from subgraph_counting import conversion
from subgraph_counting.graph import Graph
import subgraph_counting.graph as scg
from subgraph_counting.datasets import get_random_graph, get_zachary_graph

class TestGraphConversion(unittest.TestCase):
    """Unit tests"""
    def test_pyg_to_graph_undirected(self):
        """test conversion of undirected graph"""
        graph_pyg = get_random_graph(50, .2)
        graph = Graph(graph_pyg)
        self.assertFalse(graph.is_directed())
        self.assertEqual(graph.get_num_nodes(), graph_pyg.num_nodes)
        self.assertEqual(graph.get_num_edges(), graph_pyg.num_edges)
        self.assertEqual(graph.is_directed(), graph_pyg.is_directed())

    def test_pyg_to_igraph_directed(self):
        """test conversion of directed graph"""
        graph_pyg = get_random_graph(50, .2, True)
        graph = Graph(graph_pyg)
        self.assertTrue(graph.is_directed())
        self.assertEqual(graph.get_num_nodes(), graph_pyg.num_nodes)
        self.assertEqual(graph.get_num_edges(), graph_pyg.num_edges)
        self.assertEqual(graph.is_directed(), graph_pyg.is_directed())

    def test_node_attribute_conversion_default_1(self):
        """test conversion of node attributes to colors and labels.
        Default conversion, 1 dimensional attribute.
        """
        graph_pyg = get_zachary_graph()
        graph = Graph(graph_pyg, node_attribute='x')
        colors = graph.get_node_colors()
        labels = graph.get_node_labels()
        self.assertEqual(len(colors), graph_pyg.num_nodes)
        self.assertEqual(len(labels), graph_pyg.num_nodes)
        self.assertEqual(len(set(colors)), graph_pyg.num_nodes)
        self.assertEqual(len(set(labels)), graph_pyg.num_nodes)

    def test_edge_attribute_conversion_default_1(self):
        """test conversion of node attributes to colors and labels.
        Default conversion, 1 dimensional attribute."""
        graph_pyg = get_zachary_graph()
        graph = Graph(graph_pyg, edge_attribute='edge_attr')
        colors = graph.get_edge_colors()
        labels = graph.get_edge_labels()
        self.assertEqual(len(colors), graph_pyg.num_edges)
        self.assertEqual(len(labels), graph_pyg.num_edges)
        self.assertEqual(len(set(colors)), graph_pyg.num_edges)
        self.assertEqual(len(set(labels)), graph_pyg.num_edges)

    def test_node_attribute_conversion_default_3(self):
        """test conversion of node attributes to colors and labels
        Default conversion, 3 dimensional attribute."""
        graph_pyg = get_random_graph(50, .2)
        graph_pyg.x = torch.rand(graph_pyg.num_nodes, 3) # [num_nodes, num_node_features]
        graph = Graph(graph_pyg, node_attribute='x')
        colors = graph.get_node_colors()
        labels = graph.get_node_labels()
        self.assertEqual(len(colors), graph_pyg.num_nodes)
        self.assertEqual(len(labels), graph_pyg.num_nodes)
        self.assertEqual(len(set(colors)), graph_pyg.num_nodes)
        self.assertEqual(len(set(labels)), graph_pyg.num_nodes)

    def test_edge_attribute_conversion_default_10(self):
        """test conversion of node attributes to colors and labels
        Default conversion, 10 dimensional attribute."""
        graph_pyg = get_random_graph(50, .2)
        graph_pyg.edge_attr = torch.rand(graph_pyg.num_edges, 10) # [num_edges, num_edge_features]
        graph = Graph(graph_pyg, edge_attribute='edge_attr')
        colors = graph.get_edge_colors()
        labels = graph.get_edge_labels()
        self.assertEqual(len(colors), graph_pyg.num_edges)
        self.assertEqual(len(labels), graph_pyg.num_edges)
        self.assertEqual(len(set(colors)), graph_pyg.num_edges)
        self.assertEqual(len(set(labels)), graph_pyg.num_edges)

    def test_node_attribute_conversion_3(self):
        """test conversion of node attributes to colors and labels
        Custom conversion, 3 dimensional attribute."""
        graph_pyg = get_random_graph(50, .2)
        graph_pyg.x = torch.rand(graph_pyg.num_nodes, 3) # [num_nodes, num_node_features]
        graph = Graph(graph_pyg,
                      node_attribute='x',
                      node_attribute_to_color=lambda tensor: hash(tensor.numpy().data.tobytes()),
                      node_attribute_to_label=lambda tensor: str(tensor))
        colors = graph.get_node_colors()
        labels = graph.get_node_labels()
        self.assertEqual(len(colors), graph_pyg.num_nodes)
        self.assertEqual(len(labels), graph_pyg.num_nodes)
        self.assertEqual(len(set(colors)), graph_pyg.num_nodes)
        self.assertEqual(len(set(labels)), graph_pyg.num_nodes)

    def test_edge_attribute_conversion_10(self):
        """test conversion of node attributes to colors and labels
        Custom conversion, 10 dimensional attribute."""
        graph_pyg = get_random_graph(50, .2)
        graph_pyg.edge_attr = torch.rand(graph_pyg.num_edges, 10) # [num_edges, num_edge_features]
        graph = Graph(graph_pyg,
                edge_attribute='edge_attr',
                edge_attribute_to_color=lambda tensor: hash(tensor.numpy().data.tobytes()),
                edge_attribute_to_label=lambda tensor: str(tensor))
        colors = graph.get_edge_colors()
        labels = graph.get_edge_labels()
        self.assertEqual(len(colors), graph_pyg.num_edges)
        self.assertEqual(len(labels), graph_pyg.num_edges)
        self.assertEqual(len(set(colors)), graph_pyg.num_edges)
        self.assertEqual(len(set(labels)), graph_pyg.num_edges)

    def test_get_duplicate_edge_mask(self):
        graph_pyg = get_zachary_graph()
        graph = Graph(graph_pyg, True, node_attribute='x', edge_attribute='edge_attr')
        node_colors = graph.get_node_colors()
        node_labels = graph.get_node_labels()
        edge_colors = graph.get_edge_colors()
        edge_labels = graph.get_edge_labels()
        graph_ig = scg._graph_to_igraph(graph)

        node_count = 34
        edge_count = 78
        self.assertEqual(graph_ig.vcount(), node_count)
        self.assertEqual(len(node_colors), node_count)
        self.assertEqual(len(set(node_colors)), node_count)
        self.assertEqual(len(node_labels), node_count)
        self.assertEqual(len(set(node_labels)), node_count)
        self.assertEqual(graph_ig.ecount(), edge_count)
        self.assertEqual(len(edge_colors), edge_count)
        self.assertEqual(len(set(edge_colors)), edge_count)
        self.assertEqual(len(edge_labels), edge_count)
        self.assertEqual(len(set(edge_labels)), edge_count)

if __name__ == '__main__':
    # Execute unit tests.
    unittest.main()
