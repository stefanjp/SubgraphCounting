import warnings
warnings.filterwarnings("ignore", message=".*audio.*", category=UserWarning)
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
    # ranodm search
    dataset = 'ZINC-cycle-6-node-features'
    large_experiment_name = 'experiments-with-labels'
    sweep_configuration = {
        'batch_size': [64],
        'dataset': [dataset],
        'model': ['GraphConv'], 
        'epochs': [150],
        'node_level_conv_layers': [4, 6],
        'lr': [0.01],
        'hidden_size': [160],
        'dropout_conv': [True],
        'graph_level_mlp_layers': [2],
        'dropout_mlp': [True],
    }
    print('Running baseline algorithms')
    baseline({
        'batch_size': 1000,
        'dataset': dataset,
    })
    configs = get_configs(sweep_configuration)
    print(f'Running {len(configs)} configurations')
    for config in configs:
        print(f'Running following configuration: {config}')
        experiment(config, log_path=f'tb-logs-{large_experiment_name}', patience=1000)