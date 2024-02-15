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

Dependencies

Used following conda channels:
- conda-forge
- pytorch
- pyg

```
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 -c pytorch
conda install pyg -c pyg 
conda install python-igraph matplotlib -c conda-forge
conda install pytorch-lightning -c conda-forge

```
or create a new environment (might only work on Windows)
```
conda create --name <env> --file requirements.txt
```

Code was developed and tested on Windows 11 with following package versions:
* pyg==2.1.0
* python-igraph==0.10.4
* pytorch==1.12.0

## Experiments
From the project root directory execute following python files:
- python experiments/run_experiments.py
   - ZINC-star-3-star-4* experiments
     - loss
/test-feature-0 correspons to the MSE of 3-star
     - loss
/test-feature-1 correspons to the MSE of 4-star
   - ZINC-cycle-3-cycle-6* experiments
     - loss
/test-feature-0 correspons to the MSE of 3-cycle
     - loss
/test-feature-1 correspons to the MSE of 6-cycle


The baseline results as well as the model results will be logged in the tb-logs directory that can be visualized with tensorboard.