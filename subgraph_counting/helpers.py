
def get_subgraph_mask(g, g_sub, sub_vertex_list):
    subgraph_edges = set()
    subgraph_nodes = set()
    for subgraph in sub_vertex_list:
        subgraph_nodes = subgraph_nodes.union(subgraph)
        g_sub_to_g_vertex = {i: v for i, v in enumerate(subgraph)}
        for sub_edge in g_sub.es:
            subgraph_edges.add((g_sub_to_g_vertex[sub_edge.source], g_sub_to_g_vertex[sub_edge.target]))
    node_mask = [True if node.index in subgraph_nodes else False for node in g.vs]
    edge_mask = [True if edge.tuple in subgraph_edges else False for edge in g.es]
    return node_mask, edge_mask