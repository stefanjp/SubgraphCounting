"""Test pattern creation"""
import unittest
from igraph import Graph
from subgraph_counting import pattern
class TestPatternCreation(unittest.TestCase):
    """Unit tests"""
    def test_cycle(self):
        """create cycle and test isomorphism to hand crafted cycle"""
        cycle_4 = pattern.create_cycle(4)
        cycle_4_test = Graph(n=4, edges=[(0,1), (1, 2), (2, 3), (3, 0)])
        self.assertTrue(cycle_4.isomorphic(cycle_4_test))

    def test_clique(self):
        """create clique and test isomorphism to handcrafted clique"""
        clique_4 = pattern.create_clique(4)
        clique_4_test = Graph(n=4, edges=
            [
                (0,1), (0, 2), (0, 3),
                (1, 2), (1, 3),
                (2, 3)
            ])
        self.assertTrue(clique_4.isomorphic(clique_4_test))

    def test_path(self):
        """create path of length 4 and test isomorphism to handcrafted path"""
        path_4 = pattern.create_path(4)
        path_4_test = Graph(n=4,
            edges= [
                (0, 1), (1, 2), (2, 3)
            ])
        self.assertTrue(path_4.isomorphic(path_4_test))

    def test_star(self):
        """create star and test isomorphism to handcrafted path"""
        star_4 = pattern.create_star(4)
        star_4_test = Graph(n=4,
            edges= [
                (0, 1), (0, 2), (0, 3)
            ])
        self.assertTrue(star_4.isomorphic(star_4_test))

if __name__ == '__main__':
    # Execute unit tests.
    unittest.main()
