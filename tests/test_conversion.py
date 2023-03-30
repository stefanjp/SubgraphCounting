"""Test graph and attribute conversion"""
import unittest
import torch
from subgraph_counting import conversion
from subgraph_counting.datasets import get_random_graph, get_zachary_graph

class TestGraphConversion(unittest.TestCase):
    """Unit tests"""
    def test_pyg_to_igraph_undirected(self):
        """test conversion of undirected graph"""
        graph_pyg = get_random_graph(50, .2)
        graph_ig = conversion.pyg_to_igraph(graph_pyg)
        self.assertFalse(graph_ig.is_directed())
        self.assertEqual(graph_ig.vcount(), graph_pyg.num_nodes)
        self.assertEqual(graph_ig.ecount(), graph_pyg.num_edges)
        self.assertEqual(graph_ig.is_directed(), graph_pyg.is_directed())

    def test_pyg_to_igraph_directed(self):
        """test conversion of directed graph"""
        graph_pyg = get_random_graph(50, .2, True)
        graph_ig = conversion.pyg_to_igraph(graph_pyg)
        self.assertTrue(graph_ig.is_directed())
        self.assertEqual(graph_ig.vcount(), graph_pyg.num_nodes)
        self.assertEqual(graph_ig.ecount(), graph_pyg.num_edges)
        self.assertEqual(graph_ig.is_directed(), graph_pyg.is_directed())

    def test_node_attribute_conversion_default_1(self):
        """test conversion of node attributes to colors and labels.
        Default conversion, 1 dimensional attribute.
        """
        graph_pyg = get_zachary_graph()
        colors, labels = conversion.pyg_get_node_attributes(graph_pyg)
        self.assertEqual(len(colors), graph_pyg.num_nodes)
        self.assertEqual(len(labels), graph_pyg.num_nodes)
        self.assertEqual(len(set(colors)), graph_pyg.num_nodes)
        self.assertEqual(len(set(labels)), graph_pyg.num_nodes)

    def test_edge_attribute_conversion_default_1(self):
        """test conversion of node attributes to colors and labels.
        Default conversion, 1 dimensional attribute."""
        graph_pyg = get_zachary_graph()
        colors, labels = conversion.pyg_get_edge_attributes(graph_pyg)
        self.assertEqual(len(colors), graph_pyg.num_edges)
        self.assertEqual(len(labels), graph_pyg.num_edges)
        self.assertEqual(len(set(colors)), graph_pyg.num_edges)
        self.assertEqual(len(set(labels)), graph_pyg.num_edges)

    def test_node_attribute_conversion_default_3(self):
        """test conversion of node attributes to colors and labels
        Default conversion, 3 dimensional attribute."""
        graph_pyg = get_random_graph(50, .2)
        graph_pyg.x = torch.rand(graph_pyg.num_nodes, 3) # [num_nodes, num_node_features]
        colors, labels = conversion.pyg_get_node_attributes(graph_pyg)
        self.assertEqual(len(colors), graph_pyg.num_nodes)
        self.assertEqual(len(labels), graph_pyg.num_nodes)
        self.assertEqual(len(set(colors)), graph_pyg.num_nodes)
        self.assertEqual(len(set(labels)), graph_pyg.num_nodes)

    def test_edge_attribute_conversion_default_10(self):
        """test conversion of node attributes to colors and labels
        Default conversion, 10 dimensional attribute."""
        graph_pyg = get_random_graph(50, .2)
        graph_pyg.edge_attr = torch.rand(graph_pyg.num_edges, 10) # [num_edges, num_edge_features]
        colors, labels = conversion.pyg_get_edge_attributes(graph_pyg)
        self.assertEqual(len(colors), graph_pyg.num_edges)
        self.assertEqual(len(labels), graph_pyg.num_edges)
        self.assertEqual(len(set(colors)), graph_pyg.num_edges)
        self.assertEqual(len(set(labels)), graph_pyg.num_edges)

    def test_node_attribute_conversion_3(self):
        """test conversion of node attributes to colors and labels
        Custom conversion, 3 dimensional attribute."""
        graph_pyg = get_random_graph(50, .2)
        graph_pyg.x = torch.rand(graph_pyg.num_nodes, 3) # [num_nodes, num_node_features]
        colors, labels = conversion.pyg_get_node_attributes(graph_pyg,
            to_color=lambda tensor: hash(tensor.data.tobytes()),
            to_label=lambda tensor: str(tensor))
        self.assertEqual(len(colors), graph_pyg.num_nodes)
        self.assertEqual(len(labels), graph_pyg.num_nodes)
        self.assertEqual(len(set(colors)), graph_pyg.num_nodes)
        self.assertEqual(len(set(labels)), graph_pyg.num_nodes)

    def test_edge_attribute_conversion_10(self):
        """test conversion of node attributes to colors and labels
        Custom conversion, 10 dimensional attribute."""
        graph_pyg = get_random_graph(50, .2)
        graph_pyg.edge_attr = torch.rand(graph_pyg.num_edges, 10) # [num_edges, num_edge_features]
        colors, labels = conversion.pyg_get_edge_attributes(graph_pyg,
            to_color=lambda tensor: hash(tensor.data.tobytes()),
            to_label=lambda tensor: str(tensor))
        self.assertEqual(len(colors), graph_pyg.num_edges)
        self.assertEqual(len(labels), graph_pyg.num_edges)
        self.assertEqual(len(set(colors)), graph_pyg.num_edges)
        self.assertEqual(len(set(labels)), graph_pyg.num_edges)

    def test_get_duplicate_edge_mask(self):
        graph_pyg = get_zachary_graph()
        duplicate_edge_mask = conversion.pyg_get_duplicate_edge_mask(graph_pyg)
        edge_mask = ~duplicate_edge_mask
        graph_ig = conversion.pyg_to_igraph(graph_pyg, edge_mask, True)
        node_colors, node_labels = conversion.pyg_get_node_attributes(graph_pyg)
        edge_colors, edge_labels = conversion.pyg_get_edge_attributes(graph_pyg, attr_mask=edge_mask)
        
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
