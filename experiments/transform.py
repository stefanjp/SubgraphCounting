from torch_geometric.data import Data
from torch_geometric.utils.degree import degree
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from typing import Callable, Sequence
import torch
import subgraph_counting.graph as scg

@functional_transform('annotate_subgraphisomorphisms')
class AnnotateSubgraphIsomorphisms(BaseTransform):
    """Counts and annotates subgraphstructures.

    Parameters
    ----------
        subgraphs : Sequence[Data]
            Sequence of subgraphs that are counted. The annotation
            is a torch Tensor that has one entry for each subgraph
            in the sequence.
        force_undirected : bool, default=False
            If set to True, the edges of both the graph and the 
            subgraphs will be transformed to undirected graphs.
        to_undirected_reduce: str, default="mean"
            See torch_geometric.utils.undirected.to_undirected
            parameter reduce.
        node_attribute: str, optional
            If set (e.g. to 'x'), the attribute is used for using 
            code colors for subgraph matching. Both the graph in 
            the dataset and the subgraph are required to have this
            attribute.
        node_attribute_to_color: Callable[[torch.Tensor], int], optional
            Function to map torch.Tensor to node color for node 
            attribute matching. There is a default implementation
            that should always work for torch.Tensors containing a 
            single value per node. Also for more dimensional values
            there is an implementation using a hash function that 
            might yield incorrect results in some cases.
        edge_attribute: str, optional
            Equivalent to node_attribute but for edges.
        edge_attribute_to_color: Callable[[torch.Tensor], int], optional
            Equivalent to node_attribute_to_color but for edges.
        target_attribute: str, default="y"
            The value will be assigned to this graph-level attribute
            in the returned data object.

    Returns
    ----------
        Transformed data object.
    """
    def __init__(
        self,
        subgraphs: Sequence[Data],
        force_undirected: bool = False,
        to_undirected_reduce: str = "mean",
        node_attribute: str | None = None,
        node_attribute_to_color: Callable[[torch.Tensor], int] | None = None,
        edge_attribute: str | None = None,
        edge_attribute_to_color: Callable[[torch.Tensor], int] | None = None,
        target_attribute: str = "y"
    ):
        self.subgraphs = subgraphs
        self.force_undirected = force_undirected
        self.to_undirected_reduce = to_undirected_reduce
        self.node_attribute = node_attribute
        self.node_attribute_to_color = node_attribute_to_color
        self.edge_attribute = edge_attribute
        self.edge_attribute_to_color = edge_attribute_to_color
        self.target_attribute = target_attribute

    def __call__(self, data: Data) -> Data:
        graph = scg.Graph(data,
                          self.force_undirected,
                          self.to_undirected_reduce,
                          self.node_attribute,
                          self.node_attribute_to_color,
                          self.edge_attribute,
                          self.edge_attribute_to_color)
        y = []
        for sg in self.subgraphs:
            subgraph = scg.Graph(sg,
                self.force_undirected,
                self.to_undirected_reduce,
                self.node_attribute,
                self.node_attribute_to_color,
                self.edge_attribute,
                self.edge_attribute_to_color)
            subisomorphisms = graph.count_subisomorphisms(subgraph)
            y.append(subisomorphisms)
        data[self.target_attribute] = torch.as_tensor(y)
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(subgraphs={len(self.subgraphs)})'
    


@functional_transform('annotate_ones')
class AnnotateOnes(BaseTransform):
    r"""Overwrite node features with ones.
    """
    def __init__(
        self,
        node_attribute: str = 'x'
    ):
        self.node_attribute = node_attribute

    def __call__(self, data: Data) -> Data:
        data[self.node_attribute] = torch.ones_like(data[self.node_attribute])
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'

@functional_transform('shorten')
class Shorten(BaseTransform):
    r"""Throw away some node feature or target feature dimensions.
    """
    def __init__(
        self,
        node_attribute: str = 'y',
        dim=0,
        start=None,
        stop=None
    ):
        self.node_attribute = node_attribute
        self.dim = dim
        self.start = start
        self.stop = stop

    def __call__(self, data: Data) -> Data:
        dim_size = data[self.node_attribute].shape[self.dim]
        if self.start is not None and self.stop is not None:
            indices = torch.tensor(range(self.start, self.stop))
        elif self.start is not None:
            indices = torch.tensor(range(self.start, dim_size))
        elif self.stop is not None:
            indices = torch.tensor(range(self.stop))
        else:
            indices = torch.tensor(range(dim_size))
        data[self.node_attribute] = data[self.node_attribute].index_select(self.dim, indices)
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'


@functional_transform('select')
class Select(BaseTransform):
    r"""Select specific node feature or target feature dimensions and throw away the rest.
    """
    def __init__(
        self,
        node_attribute: str = 'y',
        dim=0,
        selection=[]
    ):
        self.node_attribute = node_attribute
        self.dim = dim
        self.selection = torch.as_tensor(selection, dtype=torch.int32)

    def __call__(self, data: Data) -> Data:
        data[self.node_attribute] = data[self.node_attribute].index_select(self.dim, self.selection)
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'

@functional_transform('annotate_degree')
class AnnotateDegree(BaseTransform):
    r"""Overwrite node features with ones.
    """
    def __init__(
        self,
        node_attribute: str = 'x',
        replace: bool = False,
        norm = 1
    ):
        self.node_attribute = node_attribute
        self.replace = replace
        self.norm = norm

    def __call__(self, data: Data) -> Data:
        if self.replace:
            data[self.node_attribute] = degree(data.edge_index[0], data.num_nodes).unsqueeze(1)
        else:
            in_degree = degree(data.edge_index[0], data.num_nodes).unsqueeze(1)
            data[self.node_attribute] = torch.cat([data[self.node_attribute], in_degree], dim=1)
        data[self.node_attribute] = data[self.node_attribute] * self.norm
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'


@functional_transform('norm')
class NormNodeAttribute(BaseTransform):
    r"""Overwrite node features with ones.
    """
    def __init__(
        self,
        node_attribute: str = 'x',
        norm = 1
    ):
        self.node_attribute = node_attribute
        self.norm = norm

    def __call__(self, data: Data) -> Data:
        data[self.node_attribute] = data[self.node_attribute] * self.norm
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'