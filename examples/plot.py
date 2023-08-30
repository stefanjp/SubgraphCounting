
from subgraph_counting import datasets, graph

PLOT_ROOT = "data/figures"
ZINC_DATASET = "data/datasets/ZINC/original"

def plot_cycle_in_zachary_graph():
    g = graph.Graph(
        datasets.get_zachary_graph(),
        True,
        node_attribute="x",
        edge_attribute="edge_attr",
    )
    node_font_size = 5
    edge_font_size = 3

    g.plot(
        f"{PLOT_ROOT}/_test_.svg",
        node_colors=False,
        node_labels=True,
        edge_labels=True,
        visual_style={
            "layout": "circle",
            "vertex_size": 0.1,
            "vertex_label_size": node_font_size,
            "edge_width": 0.5,
            "edge_label_size": edge_font_size,
        },
    )


if __name__ == "__main__":
    # execute graph substructure counting examples
    plot_cycle_in_zachary_graph()
    plot_star()
    plot_cycle()