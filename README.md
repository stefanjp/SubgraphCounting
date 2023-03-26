# Subgraph Counting
This repository contains code for counting subgraph structures for homogeneous and heterogeneous graphs. Also contains functionality for conversion from Pytorch Geometric Graphs to python igraph and plotting functionality. The algorithms are used from the [python-igraph](https://python.igraph.org/en/stable/) package. More specifically following algorithms are used:
* [VF2](https://python.igraph.org/en/0.10.2/api/igraph.GraphBase.html#get_subisomorphisms_vf2)
This algorith allows counting heterogeneous subgaphs. Contrains for vertex and edge matching can be specified by passing vertex/edge color information to the algorithm. If more specific constraints are required, a function can be passed which can evaluate if two vertices or edges are matching.

* [LAD](https://python.igraph.org/en/0.10.2/api/igraph.GraphBase.html#get_subisomorphisms_lad)
For homogeneous graphs, it is possible to count and get both induced and non-induced subgraphs. Vertex matching can be restricted.

## Setup
```
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 -c pytorch
conda install pyg -c pyg # tested with pyg==2.1.0
conda install python-igraph matplotlib -c conda-forge # tested with python-igraph==0.10.4
```

## Usage
TODO
