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
    dataset = 'ZINC-cycle-3-cycle-6'
    # ranodm search
    sweep_configuration = {
        'model': ['default'],
        'batch_size': [64],
        'dataset': [dataset],
        'epochs': [200],
        'lr': [0.001],
        'hidden_size': [160],
        'dropout_conv': [True],
        'graph_level_mlp_layers': [2, 4],
        'dropout_mlp': [True],
    }
    print('Running baseline algorithms')
    baseline({
        'batch_size': 1000,
        'dataset': dataset,
    })
    configs = get_configs(sweep_configuration, 2)
    print(f'Running {len(configs)} configurations')
    for config in configs:
        print(f'Running following configuration: {config}')
        experiment(config)