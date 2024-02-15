import warnings
warnings.filterwarnings("ignore", message=".*audio.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*does not have many workers which may be a bottleneck.*", category=UserWarning)

import torch
import torch_geometric as pyg
import pytorch_lightning as L

from experiments.util import get_configs
from experiments.run import baseline, experiment
if __name__ == "__main__":
    print(f"torch version {torch.__version__}")
    print(f"torch_geometric version {pyg.__version__}")
    print(f"pytorch_lightning version {L.__version__}")
    # comment out or change seed if you want to perform new sweep samples
    L.seed_everything(20)
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    large_experiment_name = 'hyperparameter-random-search'
    datasets = ['ZINC-cycle-3-cycle-6'] #, 'ZINC-star-3-star-4', 'ZINC-cycle-3-cycle-6'
    # ranodm search
    sweep_configuration = {
        'batch_size': [64, 128, 512],
        'dataset': datasets, #
        'model': ['GraphConv'], # 
        'epochs': [150],
        'lr': [.01, 0.005, 0.001],
        'node_level_conv_layers': [2, 4, 6],
        'dropout_conv': [True],
        'hidden_size': [80, 160],
        'graph_level_mlp_layers': [0, 2, 4, 6],
        'dropout_mlp': [True],
    }
    '''
    print('Running baseline algorithms')
    for dataset in datasets:
        baseline({
            'batch_size': 1000,
            'dataset': dataset,
        })
    '''
    configs = get_configs(sweep_configuration, 20)
    print(f'Running {len(configs)} configurations')
    for i, config in enumerate(configs):
        print(f'Running configuration {i+1}/{len(configs)}: {config}')
        experiment(config, log_path=f'tb-logs-{large_experiment_name}', patience=1000)