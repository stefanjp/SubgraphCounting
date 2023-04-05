# Subgraph Counting
This repository contains code for counting subgraph structures for homogeneous graphs (torch_geometric.data.Data). Also contains plotting functionality. The subgraph counting algorithms are used from the [python-igraph](https://python.igraph.org/en/stable/) package. More specifically following algorithm is used:
* [VF2](https://python.igraph.org/en/0.10.2/api/igraph.GraphBase.html#get_subisomorphisms_vf2)
This algorith allows counting subgaphs with vertex and edge constraints. Contrains for vertex and edge matching can be specified by passing vertex/edge color information to the algorithm. If more specific constraints are required, a function can be passed which can evaluate if two vertices or edges are matching.

## Setup
Clone repository
```
git clone https://github.com/stefanjp/SubgraphCounting
cd SubgraphCounting
```

Add module to PYTHONPATH
```
export PYTHONPATH=$PYTHONPATH:<PROJECT_ROOT>
```
where <PROJECT_ROOT> is the project root.

```
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 -c pytorch
conda install pyg -c pyg 
conda install python-igraph matplotlib -c conda-forge
```

Tested on Windows 11 with following package versions:
* pyg==2.1.0
* python-igraph==0.10.4
* pytorch==1.12.0

## Usage
TODO


## Contribution
