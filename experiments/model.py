from torch_geometric.nn import (
    GraphConv,
    Linear,
    BatchNorm,
    Sequential,
    global_add_pool,
    global_mean_pool,
)
import torch.nn as nn
import torch
import pytorch_lightning as L


class LightningGraphConv(L.LightningModule):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 256,
        target_size: int = 1,
        lr: float = 0.01,
        dropout_conv: bool = False,
        graph_level_mlp_layers=0,
        conv_layers: int = 3,
        dropout_mlp: bool = False,
        batch_size: int = 128,  # only for storing to hyperparameters hparams
    ):
        super().__init__()
        self.save_hyperparameters()
        sequential_tuples = []
        relu_slope = 0.1

        # adjust variance in initialization
        # https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/03-initialization-and-optimization.html
        gain_leaky = nn.init.calculate_gain("leaky_relu", relu_slope)

        for layer in range(conv_layers):
            conv = GraphConv(input_size if layer == 0 else hidden_size, hidden_size)
            # optimal gain changes for leaky relu activation.
            conv.lin_rel.weight.data.mul_(gain_leaky)
            conv.lin_root.weight.data.mul_(gain_leaky)
            sequential_tuples.append((conv, "x, edge_index -> x"))
            sequential_tuples.append((BatchNorm(hidden_size), "x -> x"))
            sequential_tuples.append((nn.LeakyReLU(relu_slope), "x -> x"))
            # reasoning for dropout percentage
            # https://stats.stackexchange.com/questions/240305/where-should-i-place-dropout-layers-in-a-neural-network
            if dropout_conv:
                sequential_tuples.append((nn.Dropout(0.1), "x -> x"))

        sequential_tuples.append((global_add_pool, "x, batch_index -> x"))
        if graph_level_mlp_layers > 0:
            for _ in range(graph_level_mlp_layers):
                linear = Linear(hidden_size, hidden_size)
                linear.weight.data.mul_(gain_leaky)
                sequential_tuples.append((linear, "x -> x"))
                sequential_tuples.append((BatchNorm(hidden_size), "x -> x"))
                sequential_tuples.append((nn.LeakyReLU(relu_slope), "x -> x"))
                # reasoning for dropout percentage
                # https://stats.stackexchange.com/questions/240305/where-should-i-place-dropout-layers-in-a-neural-network
                if dropout_mlp:
                    sequential_tuples.append((nn.Dropout(0.5), "x -> x"))

        sequential_tuples.append((Linear(hidden_size, target_size), "x -> y_predict"))

        self.model = Sequential(
            "x, edge_index, batch_index",
            sequential_tuples,
        )
        self.criterion = nn.MSELoss()
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.best_metric = []

    def forward(self, x, edge_index, batch_index):
        y_predict = self.model(x, edge_index, batch_index)
        return y_predict

    def _shared_eval_step(self, batch):
        x, y, edge_index, batch_node_indices = (
            batch.x,
            batch.y.reshape(-1, self.hparams.target_size),
            batch.edge_index,
            batch.batch,
        )
        y_hat = self.forward(x, edge_index, batch_node_indices)
        loss = self.criterion(y_hat, y)
        per_feature_loss = torch.mean(torch.square(y_hat - y), dim=0)
        return loss, per_feature_loss

    def training_step(self, batch, batch_index):
        loss, _ = self._shared_eval_step(batch)
        # metrics here
        self.log("loss/train", loss, batch_size=batch.y.shape[0])
        return loss

    def validation_step(self, batch, batch_index):
        loss, per_feature_loss = self._shared_eval_step(batch)
        if not self.trainer.sanity_checking:
            self.validation_step_outputs.append([loss, per_feature_loss])

    def test_step(self, batch, batch_index):
        loss, per_feature_loss = self._shared_eval_step(batch)
        self.test_step_outputs.append([loss, per_feature_loss])

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return
        per_feature_losses = []
        losses = []
        for loss, per_feature_loss in self.validation_step_outputs:
            losses.append(loss)
            per_feature_losses.append(per_feature_loss)
        self.best_metric.append(torch.mean(torch.stack(losses)))
        loss_per_feature = torch.mean(torch.stack(per_feature_losses), dim=0)
        for i in range(loss_per_feature.shape[0]):
            self.log(f"loss/val-feature-{i}", loss_per_feature[i].item())
        self.validation_step_outputs.clear()  # free memory
        best_metric = torch.min(torch.stack(self.best_metric)).item()
        self.log("loss/val", best_metric)

    def on_test_epoch_end(self):
        if not self.test_step_outputs:
            return
        per_feature_losses = []
        losses = []
        for loss, per_feature_loss in self.test_step_outputs:
            losses.append(loss)
            per_feature_losses.append(per_feature_loss)
        loss_per_feature = torch.mean(torch.stack(per_feature_losses), dim=0)
        for i in range(loss_per_feature.shape[0]):
            self.log(f"loss/test-feature-{i}", loss_per_feature[i].item())
        self.test_step_outputs.clear()  # free memory
        best_metric = torch.mean(torch.stack(losses)).item()
        self.log("hp_metric", best_metric)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class DoNothingOptimizer(torch.optim.Optimizer):
    def __init__(self):
        super().__init__([torch.zeros(0)], {})
        pass

    def step(self, closure):
        closure
        pass


class LightningBaseline(L.LightningModule):
    def __init__(
        self,
        mode: str,
        input_size: int = 1,
        target_size: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.y_values = []
        self.y_pred = torch.zeros(1, device=self.device)
        self.test_step_outputs = []
        self.criterion = nn.MSELoss()

    def training_step(self, batch, batch_index):
        y = batch.y.reshape(-1, self.hparams.target_size).cpu()
        self.y_values.append(y)
        return torch.zeros(1, device=self.device)

    def test_step(self, batch, batch_index):
        y = batch.y.reshape(-1, self.hparams.target_size).cpu()
        loss = self.criterion(self.y_pred, y)
        per_feature_loss = torch.mean(torch.square(self.y_pred - y), dim=0)
        if not self.trainer.sanity_checking:
            self.test_step_outputs.append([loss, per_feature_loss])

    def on_test_epoch_start(self):
        if not self.y_values:
            return
        self.y_pred = torch.mean(torch.stack(self.y_values), dim=[0, 1]).unsqueeze(0)
        if self.hparams.mode == "zero":
            self.y_pred = torch.zeros_like(self.y_pred)

    def on_test_epoch_end(self):
        if not self.test_step_outputs:
            return
        per_feature_losses = []
        losses = []
        for loss, per_feature_loss in self.test_step_outputs:
            losses.append(loss)
            per_feature_losses.append(per_feature_loss)

        val_loss_per_feature = torch.mean(torch.stack(per_feature_losses), dim=0)
        for i in range(val_loss_per_feature.shape[0]):
            self.log(f"loss/test-feature-{i}", val_loss_per_feature[i].item())
        self.log("hp_metric", torch.mean(torch.stack(losses)))
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        return DoNothingOptimizer()


from torch_geometric.nn import GIN


class LightningGIN(L.LightningModule):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 256,
        target_size: int = 1,
        lr: float = 0.01,
        conv_layers: int = 5,
        graph_level_mlp_layers: int = 2,
        dropout_conv: bool = False,
        dropout_mlp: bool = False,
        batch_size: int = 128,  # only for storing to hyperparameters hparams
    ):
        super().__init__()
        self.save_hyperparameters()
        sequential_tuples = []

        conv = GIN(
            in_channels=input_size,
            hidden_channels=hidden_size,
            num_layers=conv_layers,
            out_channels=hidden_size,
            act='LeakyReLU',
            norm='BatchNorm',
            dropout=0.1 if dropout_conv else 0,
            jk='cat'
        )
        sequential_tuples.append((conv, "x, edge_index -> x"))

        relu_slope = 0.1

        # adjust variance in initialization
        # https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/03-initialization-and-optimization.html
        gain_leaky = nn.init.calculate_gain("leaky_relu", relu_slope)
        sequential_tuples.append((global_add_pool, "x, batch_index -> x"))
        if graph_level_mlp_layers > 0:
            for _ in range(graph_level_mlp_layers):
                linear = Linear(hidden_size, hidden_size)
                linear.weight.data.mul_(gain_leaky)
                sequential_tuples.append((linear, "x -> x"))
                sequential_tuples.append((BatchNorm(hidden_size), "x -> x"))
                sequential_tuples.append((nn.LeakyReLU(relu_slope), "x -> x"))
                # reasoning for dropout percentage
                # https://stats.stackexchange.com/questions/240305/where-should-i-place-dropout-layers-in-a-neural-network
                if dropout_mlp:
                    sequential_tuples.append((nn.Dropout(0.5), "x -> x"))

        sequential_tuples.append((Linear(hidden_size, target_size), "x -> y_predict"))

        self.model = Sequential(
            "x, edge_index, batch_index",
            sequential_tuples,
        )

        self.criterion = nn.MSELoss()
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.best_metric = []

    def forward(self, x, edge_index, batch_index):
        y_predict = self.model(x, edge_index, batch_index)
        return y_predict

    def _shared_eval_step(self, batch):
        x, y, edge_index, batch_node_indices = (
            batch.x,
            batch.y.reshape(-1, self.hparams.target_size),
            batch.edge_index,
            batch.batch,
        )
        y_hat = self.forward(x, edge_index, batch_node_indices)
        loss = self.criterion(y_hat, y)
        per_feature_loss = torch.mean(torch.square(y_hat - y), dim=0)
        return loss, per_feature_loss

    def training_step(self, batch, batch_index):
        loss, _ = self._shared_eval_step(batch)
        # metrics here
        self.log("loss/train", loss, batch_size=batch.y.shape[0])
        return loss

    def validation_step(self, batch, batch_index):
        loss, per_feature_loss = self._shared_eval_step(batch)
        if not self.trainer.sanity_checking:
            self.validation_step_outputs.append([loss, per_feature_loss])

    def test_step(self, batch, batch_index):
        loss, per_feature_loss = self._shared_eval_step(batch)
        self.test_step_outputs.append([loss, per_feature_loss])

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return
        per_feature_losses = []
        losses = []
        for loss, per_feature_loss in self.validation_step_outputs:
            losses.append(loss)
            per_feature_losses.append(per_feature_loss)
        self.best_metric.append(torch.mean(torch.stack(losses)))
        loss_per_feature = torch.mean(torch.stack(per_feature_losses), dim=0)
        for i in range(loss_per_feature.shape[0]):
            self.log(f"loss/val-feature-{i}", loss_per_feature[i].item())
        self.validation_step_outputs.clear()  # free memory
        best_metric = torch.min(torch.stack(self.best_metric)).item()
        self.log("loss/val", best_metric)

    def on_test_epoch_end(self):
        if not self.test_step_outputs:
            return
        per_feature_losses = []
        losses = []
        for loss, per_feature_loss in self.test_step_outputs:
            losses.append(loss)
            per_feature_losses.append(per_feature_loss)
        loss_per_feature = torch.mean(torch.stack(per_feature_losses), dim=0)
        for i in range(loss_per_feature.shape[0]):
            self.log(f"loss/test-feature-{i}", loss_per_feature[i].item())
        self.test_step_outputs.clear()  # free memory
        best_metric = torch.mean(torch.stack(losses)).item()
        self.log("hp_metric", best_metric)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

from torch_geometric.nn import GAT
class LightningGATv2(L.LightningModule):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 256,
        target_size: int = 1,
        lr: float = 0.01,
        conv_layers: int = 5,
        graph_level_mlp_layers: int = 2,
        dropout_conv: bool = False,
        dropout_mlp: bool = False,
        batch_size: int = 128,  # only for storing to hyperparameters hparams
    ):
        super().__init__()
        self.save_hyperparameters()
        sequential_tuples = []
        conv = GAT(
            in_channels=input_size,
            hidden_channels=hidden_size,
            num_layers=conv_layers,
            out_channels=hidden_size,
            act='LeakyReLU',
            norm='BatchNorm',
            dropout=0.1 if dropout_conv else 0,
            v2=True,
            jk='cat'
        )
        sequential_tuples.append((conv, "x, edge_index -> x"))

        relu_slope = 0.1

        # adjust variance in initialization
        # https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/03-initialization-and-optimization.html
        gain_leaky = nn.init.calculate_gain("leaky_relu", relu_slope)
        sequential_tuples.append((global_add_pool, "x, batch_index -> x"))
        if graph_level_mlp_layers > 0:
            for _ in range(graph_level_mlp_layers):
                linear = Linear(hidden_size, hidden_size)
                linear.weight.data.mul_(gain_leaky)
                sequential_tuples.append((linear, "x -> x"))
                sequential_tuples.append((BatchNorm(hidden_size), "x -> x"))
                sequential_tuples.append((nn.LeakyReLU(relu_slope), "x -> x"))
                # reasoning for dropout percentage
                # https://stats.stackexchange.com/questions/240305/where-should-i-place-dropout-layers-in-a-neural-network
                if dropout_mlp:
                    sequential_tuples.append((nn.Dropout(0.5), "x -> x"))

        sequential_tuples.append((Linear(hidden_size, target_size), "x -> y_predict"))

        self.model = Sequential(
            "x, edge_index, batch_index",
            sequential_tuples,
        )
        self.criterion = nn.MSELoss()
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.best_metric = []

    def forward(self, x, edge_index, batch_index):
        y_predict = self.model(x, edge_index, batch_index)
        return y_predict

    def _shared_eval_step(self, batch):
        x, y, edge_index, batch_node_indices = (
            batch.x,
            batch.y.reshape(-1, self.hparams.target_size),
            batch.edge_index,
            batch.batch,
        )
        y_hat = self.forward(x, edge_index, batch_node_indices)
        loss = self.criterion(y_hat, y)
        per_feature_loss = torch.mean(torch.square(y_hat - y), dim=0)
        return loss, per_feature_loss

    def training_step(self, batch, batch_index):
        loss, _ = self._shared_eval_step(batch)
        # metrics here
        self.log("loss/train", loss, batch_size=batch.y.shape[0])
        return loss

    def validation_step(self, batch, batch_index):
        loss, per_feature_loss = self._shared_eval_step(batch)
        if not self.trainer.sanity_checking:
            self.validation_step_outputs.append([loss, per_feature_loss])

    def test_step(self, batch, batch_index):
        loss, per_feature_loss = self._shared_eval_step(batch)
        self.test_step_outputs.append([loss, per_feature_loss])

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return
        per_feature_losses = []
        losses = []
        for loss, per_feature_loss in self.validation_step_outputs:
            losses.append(loss)
            per_feature_losses.append(per_feature_loss)
        self.best_metric.append(torch.mean(torch.stack(losses)))
        loss_per_feature = torch.mean(torch.stack(per_feature_losses), dim=0)
        for i in range(loss_per_feature.shape[0]):
            self.log(f"loss/val-feature-{i}", loss_per_feature[i].item())
        self.validation_step_outputs.clear()  # free memory
        best_metric = torch.min(torch.stack(self.best_metric)).item()
        self.log("loss/val", best_metric)

    def on_test_epoch_end(self):
        if not self.test_step_outputs:
            return
        per_feature_losses = []
        losses = []
        for loss, per_feature_loss in self.test_step_outputs:
            losses.append(loss)
            per_feature_losses.append(per_feature_loss)
        loss_per_feature = torch.mean(torch.stack(per_feature_losses), dim=0)
        for i in range(loss_per_feature.shape[0]):
            self.log(f"loss/test-feature-{i}", loss_per_feature[i].item())
        self.test_step_outputs.clear()  # free memory
        best_metric = torch.mean(torch.stack(losses)).item()
        self.log("hp_metric", best_metric)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    


from torch_geometric.nn import GraphSAGE, PNA, JumpingKnowledge, GAE

class LightningOthers(L.LightningModule):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 256,
        target_size: int = 1,
        lr: float = 0.01,
        conv_layers: int = 5,
        graph_level_mlp_layers: int = 2,
        dropout_conv: bool = False,
        dropout_mlp: bool = False,
        batch_size: int = 128,  # only for storing to hyperparameters hparams
        model:str = "GraphSAGE"
    ):
        super().__init__()
        self.save_hyperparameters()
        sequential_tuples = []

        Conv = None

        if model == "GraphSAGE":
            Conv = GraphSAGE
        elif model == "PNA":
            Conv = PNA
        elif model == "JumpingKnowledge":
            Conv = JumpingKnowledge
        elif model == "GAE":
            Conv = GAE
        else:
            raise Exception("model unknown")
        
        conv = Conv(
            in_channels=input_size,
            hidden_channels=hidden_size,
            num_layers=conv_layers,
            out_channels=hidden_size,
            act='LeakyReLU',
            norm='BatchNorm',
            dropout=0.1 if dropout_conv else 0,
            jk='cat'
        )
        sequential_tuples.append((conv, "x, edge_index -> x"))

        relu_slope = 0.1

        # adjust variance in initialization
        # https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/03-initialization-and-optimization.html
        gain_leaky = nn.init.calculate_gain("leaky_relu", relu_slope)
        sequential_tuples.append((global_add_pool, "x, batch_index -> x"))
        if graph_level_mlp_layers > 0:
            for _ in range(graph_level_mlp_layers):
                linear = Linear(hidden_size, hidden_size)
                linear.weight.data.mul_(gain_leaky)
                sequential_tuples.append((linear, "x -> x"))
                sequential_tuples.append((BatchNorm(hidden_size), "x -> x"))
                sequential_tuples.append((nn.LeakyReLU(relu_slope), "x -> x"))
                # reasoning for dropout percentage
                # https://stats.stackexchange.com/questions/240305/where-should-i-place-dropout-layers-in-a-neural-network
                if dropout_mlp:
                    sequential_tuples.append((nn.Dropout(0.5), "x -> x"))

        sequential_tuples.append((Linear(hidden_size, target_size), "x -> y_predict"))

        self.model = Sequential(
            "x, edge_index, batch_index",
            sequential_tuples,
        )
        self.criterion = nn.MSELoss()
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.best_metric = []

    def forward(self, x, edge_index, batch_index):
        y_predict = self.model(x, edge_index, batch_index)
        return y_predict

    def _shared_eval_step(self, batch):
        x, y, edge_index, batch_node_indices = (
            batch.x,
            batch.y.reshape(-1, self.hparams.target_size),
            batch.edge_index,
            batch.batch,
        )
        y_hat = self.forward(x, edge_index, batch_node_indices)
        loss = self.criterion(y_hat, y)
        per_feature_loss = torch.mean(torch.square(y_hat - y), dim=0)
        return loss, per_feature_loss

    def training_step(self, batch, batch_index):
        loss, _ = self._shared_eval_step(batch)
        # metrics here
        self.log("loss/train", loss, batch_size=batch.y.shape[0])
        return loss

    def validation_step(self, batch, batch_index):
        loss, per_feature_loss = self._shared_eval_step(batch)
        if not self.trainer.sanity_checking:
            self.validation_step_outputs.append([loss, per_feature_loss])

    def test_step(self, batch, batch_index):
        loss, per_feature_loss = self._shared_eval_step(batch)
        self.test_step_outputs.append([loss, per_feature_loss])

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return
        per_feature_losses = []
        losses = []
        for loss, per_feature_loss in self.validation_step_outputs:
            losses.append(loss)
            per_feature_losses.append(per_feature_loss)
        self.best_metric.append(torch.mean(torch.stack(losses)))
        loss_per_feature = torch.mean(torch.stack(per_feature_losses), dim=0)
        for i in range(loss_per_feature.shape[0]):
            self.log(f"loss/val-feature-{i}", loss_per_feature[i].item())
        self.validation_step_outputs.clear()  # free memory
        best_metric = torch.min(torch.stack(self.best_metric)).item()
        self.log("loss/val", best_metric)

    def on_test_epoch_end(self):
        if not self.test_step_outputs:
            return
        per_feature_losses = []
        losses = []
        for loss, per_feature_loss in self.test_step_outputs:
            losses.append(loss)
            per_feature_losses.append(per_feature_loss)
        loss_per_feature = torch.mean(torch.stack(per_feature_losses), dim=0)
        for i in range(loss_per_feature.shape[0]):
            self.log(f"loss/test-feature-{i}", loss_per_feature[i].item())
        self.test_step_outputs.clear()  # free memory
        best_metric = torch.mean(torch.stack(losses)).item()
        self.log("hp_metric", best_metric)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)