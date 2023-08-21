import os

import torch
from torch_geometric.loader import DataLoader
import pytorch_lightning as L
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from experiments.model import LightningBaseline, LightningGraphConv
from experiments.util import get_dataset


def baseline(config, log_path="tb-logs"):
    datasets_pyg = get_dataset(config["dataset"])
    task_name = f"{config['dataset']}-baseline"
    # if True, directory with baseline already exists, do not calculate again
    if os.path.isdir(os.path.join(log_path, task_name)):
        return
    batch_size = config["batch_size"]
    train_loader = DataLoader(datasets_pyg["train"], batch_size, shuffle=True)
    val_loader = DataLoader(datasets_pyg["val"], batch_size)
    test_loader = DataLoader(datasets_pyg["test"], batch_size)
    node_feature_size = next(iter(datasets_pyg["train"])).x.shape[1]
    target_features_size = next(iter(datasets_pyg["train"])).y.shape[0]
    # statistics
    targets = [data.y for data in datasets_pyg["train"]]
    features = [torch.max(data.x, dim=0).values for data in datasets_pyg["train"]]

    print(f"Graphs: {len(targets)}")
    print(f"Max Target per feature: {torch.max(torch.stack(targets), dim=0).values}")
    print(f"Avg Target per feature: {torch.mean(torch.stack(targets), dim=0)}")
    print(f"Avg Target: {torch.mean(torch.stack(targets))}")
    print(f"Max Feature: {torch.max(torch.stack(features))}")

    # model_zero = LightningBaseline('zero', node_feature_size, target_features_size)
    # trainer = L.Trainer(
    #    max_epochs=1,
    #    logger=TensorBoardLogger(log_path, name=task_name),
    # )
    # trainer.fit(model_zero, train_loader, val_loader)
    # trainer.test(model_mean, test_loader)

    model_mean = LightningBaseline("mean", node_feature_size, target_features_size)
    trainer = L.Trainer(
        max_epochs=1,
        logger=TensorBoardLogger(log_path, name=task_name),
    )
    trainer.fit(model_mean, train_loader, val_loader)
    trainer.test(model_mean, test_loader)


def experiment(config, log_path="tb-logs", patience=10):
    # each experiment should be reproducible
    L.seed_everything(40)
    # If called by wandb.agent, as below,
    # this config will be set by Sweep Controller
    datasets_pyg = get_dataset(config["dataset"])
    task_name = config["dataset"]
    batch_size = config["batch_size"]
    train_loader = DataLoader(datasets_pyg["train"], batch_size, shuffle=True)
    val_loader = DataLoader(datasets_pyg["val"], batch_size)
    test_loader = DataLoader(datasets_pyg["test"], batch_size)
    node_feature_size = next(iter(datasets_pyg["train"])).x.shape[1]
    target_features_size = next(iter(datasets_pyg["train"])).y.shape[0]
    ## train/validate
    epochs = config["epochs"]
    lr = config["lr"]
    hidden_size = config["hidden_size"]
    dropout_conv = config["dropout_conv"]
    dropout_mlp = config["dropout_mlp"]
    graph_level_mlp_layers = config["graph_level_mlp_layers"]

    model = LightningGraphConv(
        target_size=target_features_size,
        input_size=node_feature_size,
        hidden_size=hidden_size,
        lr=lr,
        graph_level_mlp_layers=graph_level_mlp_layers,
        dropout_conv=dropout_conv,
        dropout_mlp=dropout_mlp,
        batch_size=batch_size,
    )
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{model}")
    print(f"Total trainable parameters: {trainable_parameters}")

    val_check_interval = len(train_loader)
    logger = TensorBoardLogger(log_path, name=task_name)
    trainer = L.Trainer(
        max_epochs=epochs,
        val_check_interval=val_check_interval,
        accelerator="gpu",
        logger=logger,
        log_every_n_steps=10,
        callbacks=[
            ModelCheckpoint(
                monitor="loss/val",
                save_on_train_epoch_end=False,
                save_last=True,
            ),
             # EarlyStopping patience: number of validation epochs without improvement
            EarlyStopping(
                monitor="loss/val", patience=patience
            ),  
        ],
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader, ckpt_path='best')
