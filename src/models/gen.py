from typing import List

import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.conv import GENConv


class GENEncoder(nn.Module):
    def __init__(
        self,
        d_features: int,
        d_edges: int,
        d_out: int,
        d_hidden_dim: int = 300,
        num_layers: int = 3,
        d_linear_layers: List[int] = [
            256,
        ],
        dropout: float = 0.01,
        activation: str = "ReLU",
    ):
        super(GENEncoder, self).__init__()
        self.num_node_features = d_features
        self.d_edges = d_edges
        self.nout = d_out
        self.d_hidden_dim = d_hidden_dim
        self.num_layers = num_layers
        self.d_linear_layers = d_linear_layers
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        self.conv = GENConv(
            in_channels=d_features,
            out_channels=d_hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            act=activation,
            edge_dim=d_edges,
        )

        self.linear_layers = [nn.Linear(d_hidden_dim, d_linear_layers[0])]
        for i in range(1, len(d_linear_layers)):
            self.linear_layers.append(
                nn.Linear(d_linear_layers[i - 1], d_linear_layers[i])
            )
        self.linear_layers.append(nn.Linear(d_linear_layers[-1], d_out))
        self.linear_layers = nn.ModuleList(self.linear_layers)

        if isinstance(activation, str):
            self.activation = getattr(nn, activation)()
        else:
            self.activation = activation

    def forward(self, batch):
        output = self.conv(batch.x, batch.edge_index, edge_attr=batch.edge_attr)

        for i, layer in enumerate(self.linear_layers):
            output = layer(output)
            if i < len(self.linear_layers) - 1:
                output = self.activation(output)
            output = self.dropout(output)

        return global_mean_pool(output, batch.batch)
