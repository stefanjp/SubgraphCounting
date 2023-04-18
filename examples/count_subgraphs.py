
def count_cycles_in_zinc_dataset():
    """Execute code that counts cycles of size 8 in all graphs of ZINC dataset.
    The ZINC dataset is interpreted as undirected graphs without vertex or edge attributes.
    """
    # get ZINC dataset
    dataset_pyg = datasets.get_zinc_dataset()['train']
    pattern_size = 10
    subgraph_pattern = pattern.create_cycle(pattern_size)
    pattern_counts = {}
    for i, graph_pyg in enumerate(dataset_pyg):
        graph_ig = conversion.pyg_to_igraph(graph_pyg, to_undirected=True)
        pattern_counts[i] = graph_ig.count_subisomorphisms_vf2(subgraph_pattern)
        if i % 200 == 0:
            print(f'Progress: {i / len(dataset_pyg) * 100:.1f}%')
    with open(f'{DATA_ZINC}/train_cycle_{pattern_size}.json', 'w', encoding ='utf8') as json_file:
        json.dump(pattern_counts, json_file, ensure_ascii = True)

def count_cycles_with_attributes_in_zinc_dataset():
    """Execute code that counts subgraph structure with attributres"""
    # get ZINC dataset
    dataset_pyg = datasets.get_zinc_dataset()['train']
    subgraph_pattern = pattern.create_cycle(6)
    subgraph_node_colors = [0 for i in range(6)]
    subgraph_edge_colors = [1, 2, 1, 2, 1, 2]
    for i, graph_pyg in enumerate(dataset_pyg):
        graph_ig = conversion.pyg_to_igraph(graph_pyg)
        node_color, _ = conversion.pyg_node_attributes(graph_pyg)
        edge_color, _ = conversion.pyg_edge_attributes(graph_pyg)
        edges = list(zip(graph_pyg.edge_index[0].tolist(), graph_pyg.edge_index[1].tolist()))
        edge_color = [graph_pyg.edge_attr[i].item() for i, index in enumerate(edges) if index[0] < index[1]]
        fig, _ = plot.plot_graph(graph_ig, f'{PLOT_ROOT}/zink_train_{i}_attributes.svg',
            node_attributes=node_color,
            edge_labels=[str(ec) for ec in edge_color])
        subisomorphisms = graph_ig.get_subisomorphisms_vf2(subgraph_pattern,
            color1=node_color, color2=subgraph_node_colors,
            edge_color1=edge_color, edge_color2=subgraph_edge_colors)

        if subisomorphisms:
            sub_nodes, sub_edges = helpers.get_subgraph_mask(graph_ig, subgraph_pattern, subisomorphisms)
            fig, _ = plot.plot_graph(graph_ig, f'{PLOT_ROOT}/zink_train_{i}_cycle_8.svg',
                    subgraph_node_mask=sub_nodes,
                    subgraph_edge_mask=sub_edges)
            plt.close(fig)

def plot_cycle_in_zachary_graph():
    graph_pyg = get_zachary_graph()
    duplicate_edge_mask = conversion.pyg_get_duplicate_edge_mask(graph_pyg)
    edge_mask = ~duplicate_edge_mask
    graph_ig = conversion.pyg_to_igraph(graph_pyg, edge_mask, True)
    _, node_attribute_labels = conversion.pyg_get_node_attributes(graph_pyg)
    _, edge_attribute_labels = conversion.pyg_get_edge_attributes(graph_pyg, attr_mask=edge_mask)
    from subgraph_counting import plot
    node_font_size = 5
    edge_font_size = 3
    plot.plot_graph(graph_ig, "test.svg", node_labels=node_attribute_labels, edge_labels=edge_attribute_labels,
                    visual_style={"layout": 'circle', "vertex_size": .1, "vertex_label_size": node_font_size,
                                    "edge_width": .5, "edge_label_size": edge_font_size})

    conversion.pyg_get_duplicate_edge_mask(graph_pyg)

if __name__ == '__main__':
    # execute graph substructure counting examples
    count_cycles_with_attributes_in_zinc_dataset()
