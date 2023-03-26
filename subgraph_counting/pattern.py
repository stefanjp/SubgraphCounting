"""
This module counts subgraph structures in a graph with conventional algorithms.
"""
from igraph import Graph
from itertools import combinations

def create_cycle(vertex_count: int) -> Graph:
    """ Creates an undirected cycle.

    Parameters
    ----------
    vertex_count : int
        number of nodes in the cycle.
        cycle: all vertices are connected in a closed chain.

    Returns
    ----------
    Undirected cycle.
    """
    edges = list(zip(range(vertex_count-1), range(1,vertex_count)))
    edges.append((vertex_count-1, 0))
    return Graph(n=vertex_count, edges=edges)

def create_path(vertex_count: int):
    """ Creates an undirected path.

    Parameters
    ----------
    vertex_count : int
        number of nodes in the path.
        path: all vertices are connected in an open chain
    
    Returns
    ----------
    Undirected path.
    """
    edges = list(zip(range(vertex_count-1), range(1,vertex_count)))
    return Graph(n=vertex_count, edges=edges)

def create_clique(vertex_count: int):
    """ Creates an undirected clique.

    Parameters
    ----------
    vertex_count : int
        number of nodes in the clique structure
        clique: all pair of nodes are connected by an edge.

    Returns
    ----------
    Undirected clique.
    """
    vertices = list(range(vertex_count))
    edges = list(combinations(vertices, 2))
    return Graph(n=vertex_count, edges=edges)

def create_star(vertex_count: int):
    """ Creates an undirected star.

    Parameters
    ----------
    vertex_count : int
        number of nodes in the star structure.
        star: a central vertex is connected to all other vertices.

    Returns
    ----------
    Undirected star.
    """
    edges = [(0, x + 1) for x in range(vertex_count - 1)]
    return Graph(n=vertex_count, edges=edges)
