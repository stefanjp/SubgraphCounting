from torch_geometric.nn import GraphConv, Linear, BatchNorm, Sequential, global_add_pool, global_mean_pool
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
        conv_layers = 3

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


### GIN convolution along the graph structure
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
import numpy as np
def center_pool(x, node_to_subgraph):
    node_to_subgraph = node_to_subgraph.cpu().numpy()
    # the first node of each subgraph is its center
    _, center_indices = np.unique(node_to_subgraph, return_index=True)
    # x = x[center_indices]
    return x[center_indices]

class GINConv(MessagePassing):
    def __init__(self, M_in, M_out):
        '''
            emb_dim (int): node embedding dimensionality
        '''
        super().__init__(aggr="add")

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(M_in, 2 * M_in),
            torch.nn.BatchNorm1d(2 * M_in),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * M_in, M_out)
        )
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.edge_encoder = torch.nn.Embedding(5, M_in)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.edge_encoder(edge_attr)
        out = self.mlp(
            (1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding)
        )
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out
    
class LightningI2GNN(L.LightningModule):
    def __init__(
        self,
        target_size:int,
        lr:float,
        num_layers:int=5,
        subgraph_pooling:str="mean",
        subgraph2_pooling:str="mean",
        use_pooling_nn:bool=False,
        use_pos:bool=False,
        use_virtual_node:bool=False,
        edge_attr_dim:int=5,
        use_rd:bool=False,
        RNI:bool=False,
        drop_ratio:float=0,
        degree:int|None=None,
        double_pooling:bool=False,
        gate:bool=False,
        batch_size: int = 128,  # only for storing to hyperparameters hparams
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        assert (subgraph_pooling=='mean' or subgraph_pooling=='add' or subgraph_pooling=='mean-context') \
               and (subgraph2_pooling=='mean' or subgraph2_pooling=='add' or subgraph2_pooling=='center' or
                    subgraph2_pooling=='mean-center' or subgraph2_pooling=='mean-center-side')
        self.subgraph_pooling = subgraph_pooling
        self.subgraph2_pooling = subgraph2_pooling
        s2_dim = 2 if subgraph2_pooling=='mean-center' else 3 if subgraph2_pooling=='mean-center-side' else 1
        s2_dim = s2_dim + 1 if subgraph_pooling == 'mean-context' else s2_dim
        self.use_rd = use_rd
        if self.use_rd:
            self.rd_projection = nn.Linear(2, 8)

        # self.z_embedding = torch.nn.Embedding(1000, 8)
        self.node_type_embedding = nn.Embedding(100, 8)
        self.double_pooling = double_pooling
        self.res = True
        self.gate = gate
        if self.gate:
            self.subgraph_gate = nn.ModuleList()

        # integer node label feature
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.z_embedding_list = nn.ModuleList()
        if self.use_rd:
            self.rd_projection_list = nn.ModuleList()

        if self.double_pooling:
            self.double_nns = nn.ModuleList()
        M_in, M_out = 1 + 8, 64
        # M_in, M_out = dataset.num_features + 8, 1


        # first layer
        self.convs.append(GINConv(2*M_in, M_out))
        self.z_embedding_list.append(nn.Embedding(100, M_in))
        if self.use_rd:
            self.rd_projection_list.append(nn.Linear(2, M_in))
        if self.gate:
            self.subgraph_gate.append(Sequential(Linear(M_in, M_out), nn.Sigmoid()))
        self.norms.append(nn.BatchNorm1d(M_out))
        if self.double_pooling:
            seq = nn.Sequential(Linear(M_out*(1+s2_dim), 128), nn.ReLU(), Linear(128, M_out))
            self.double_nns.append(seq)

        for i in range(num_layers - 1):
            # convolutional layer
            M_in, M_out = M_out, 64
            # additional distance embedding
            self.z_embedding_list.append(torch.nn.Embedding(100, M_in))
            if self.use_rd:
                self.rd_projection_list.append(torch.nn.Linear(2, M_in))
            if self.gate:
                self.subgraph_gate.append(nn.Sequential(Linear(M_in, M_out), torch.nn.Sigmoid()))

            self.convs.append(GINConv(2*M_in, M_out))
            self.norms.append(torch.nn.BatchNorm1d(M_out))

            if self.double_pooling:
                seq = nn.Sequential(Linear(M_out * (1 + s2_dim), 128), nn.ReLU(), Linear(128, M_out))
                self.double_nns.append(seq)

        # MLPs for hierarchical pooling
        if use_pooling_nn:
            self.edge_pooling_nn = nn.Sequential(Linear(s2_dim * M_out, s2_dim * M_out), nn.ReLU(),
                                              Linear(s2_dim * M_out, s2_dim * M_out))
            self.node_pooling_nn = nn.Sequential(Linear(s2_dim * M_out, s2_dim * M_out),
                                              nn.ReLU(), Linear(s2_dim * M_out, s2_dim * M_out))
        self.use_pooling_nn = use_pooling_nn

        # final graph pooling
        self.z_embedding_list.append(nn.Embedding(100, M_out))
        if self.use_rd:
            self.rd_projection_list.append(nn.Linear(2, M_out))
        if self.gate:
            self.subgraph_gate.append(nn.Sequential(Linear(M_out, M_out), nn.Sigmoid()))

        self.fc1 = nn.Linear(s2_dim * M_out, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, target_size)
    
    def graph_pooling(self, x, data, z=None, layer=None, aggr='mean', node_emb_only=False):
        if self.subgraph_pooling == 'mean-context':
            x_node = global_mean_pool(x, data.node_to_original_node)
        # subgraph2-level pooling
        if self.subgraph2_pooling == 'mean':
            if self.gate:
                x = self.subgraph_gate[layer](z) * x
            x = global_mean_pool(x, data.node_to_subgraph2)
        elif self.subgraph2_pooling == 'add':
            x = global_add_pool(x, data.node_to_subgraph2)
        elif self.subgraph2_pooling == 'center':
            if 'center_idx' in data:
                x = x[data.center_idx[:, 0]]
            else:
                x = center_pool(x, data.node_to_subgraph2)
        elif self.subgraph2_pooling == 'mean-center':
            x = torch.cat([global_mean_pool(x, data.node_to_subgraph2), center_pool(x, data.node_to_subgraph2)],
                          dim=-1)
        elif self.subgraph2_pooling == 'mean-center-side':
            if self.gate:
                x = self.subgraph_gate[layer](z) * x
            x = torch.cat([global_mean_pool(x, data.node_to_subgraph2), x[data.center_idx[:, 0]],
                           x[data.center_idx[:, 1]]], dim=-1)

        if self.use_pooling_nn:
            x = self.edge_pooling_nn(x)

        # subgraph-level pooling
        if self.subgraph_pooling == 'mean':
            x = global_mean_pool(x, data.subgraph2_to_subgraph)
        elif self.subgraph_pooling == 'add':
            x = global_add_pool(x, data.subgraph2_to_subgraph)
        elif self.subgraph_pooling == 'mean-context':
            x = torch.cat([global_mean_pool(x, data.subgraph2_to_subgraph),
                           x_node], dim=-1)

        # return node embedding
        if node_emb_only:
            return x

        if self.use_pooling_nn:
            x = self.node_pooling_nn(x)

        # subgraph to graph
        if aggr == 'mean':
            return global_mean_pool(x, data.subgraph_to_graph)
        elif aggr == 'add':
            return global_add_pool(x, data.subgraph_to_graph)

    def forward(self, data):
        data.x = data.x.type(torch.int64)
        batch = data.batch
        # integer node type embedding
        x = self.node_type_embedding(data.x)

        # concatenate with continuous node features
        x = torch.cat([x, data.x.view(-1, 1)], -1)

        # x0 = x
        for layer, conv in enumerate(self.convs):
            # distance embedding
            z_emb = self.z_embedding_list[layer](data.z)
            if z_emb.ndim == 3:
                z_emb = z_emb.sum(dim=1)
            if self.use_rd:
                z_emb = z_emb + self.rd_projection_list[layer](data.rd)
            x = torch.cat([x, z_emb], dim=-1)

            # convolution layer
            x = conv(x, data.edge_index, data.edge_attr)
            ## concat along subgraphs
            if self.double_pooling:
                x = torch.cat([x, self.graph_pooling(x, data, z_emb, layer, node_emb_only=True)[data.node_to_original_node]], dim=-1)
                x = self.double_nns[layer](x)

            x = self.norms[layer](x)
            if layer < len(self.convs) - 1:
                x = F.elu(x)
            # x = F.dropout(x, self.drop_ratio, training = self.training)

            # residual connection
            if layer > 0 and self.res:
                 x = x + x0
            x0 = x

        # graph pooling
        # distance embedding
        z_emb = self.z_embedding_list[-1](data.z)
        if z_emb.ndim == 3:
            z_emb = z_emb.sum(dim=1)
        if self.use_rd:
            z_emb = z_emb + self.rd_projection_list[-1](data.rd)

        x = self.graph_pooling(x, data, z_emb, -1)

        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        # x = F.log_softmax(x, dim=-1)
        return x
    def _shared_eval_step(self, batch):
        y = batch.y.reshape(-1, self.hparams.target_size)
        y_hat = self.forward(batch)
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