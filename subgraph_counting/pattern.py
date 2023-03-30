"""
This module counts subgraph structures in a graph with conventional algorithms.
"""
from igraph import Graph
from itertools import combinations

def create_cycle(node_count: int, duplicate_edges = False) -> Graph:
    """ Creates an undirected cycle.

    Parameters
    ----------
    node_count : int
        number of nodes in the cycle.
        cycle: all vertices are connected in a closed chain.

    Returns
    ----------
    Undirected cycle.
    """
    edges = list(zip(range(node_count-1), range(1,node_count)))
    edges.append((node_count-1, 0))
    if duplicate_edges:
        edges = edges + [(edge[1], edge[0]) for edge in edges]
    return Graph(n=node_count, edges=edges)

def create_path(node_count: int):
    """ Creates an undirected path.

    Parameters
    ----------
    node_count : int
        number of nodes in the path.
        path: all vertices are connected in an open chain
    
    Returns
    ----------
    Undirected path.
    """
    edges = list(zip(range(node_count-1), range(1,node_count)))
    return Graph(n=node_count, edges=edges)

def create_clique(node_count: int):
    """ Creates an undirected clique.

    Parameters
    ----------
    node_count : int
        number of nodes in the clique structure
        clique: all pair of nodes are connected by an edge.

    Returns
    ----------
    Undirected clique.
    """
    vertices = list(range(node_count))
    edges = list(combinations(vertices, 2))
    return Graph(n=node_count, edges=edges)

def create_star(node_count: int):
    """ Creates an undirected star.

    Parameters
    ----------
    node_count : int
        number of nodes in the star structure.
        star: a central node is connected to all other vertices.

    Returns
    ----------
    Undirected star.
    """
    edges = [(0, x + 1) for x in range(node_count - 1)]
    return Graph(n=node_count, edges=edges)
