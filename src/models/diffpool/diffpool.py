from typing import List

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool

from .encoder import GATEncoder
from .utils import batch_diffpool, extract_blocks


class DiffPoolEncoder(nn.Module):
    def __init__(
        self,
        d_features: int,
        d_out: int,
        d_pooling_layers: List[int],
        d_encoder_hidden_dims: List[int],
        d_encoder_linear_layers: List[List[int]],
        d_encoder_num_heads: List[int],
        d_encoder_num_layers: List[int],
        d_linear: int,
        dropout: float,
    ):
        super(DiffPoolEncoder, self).__init__()
        self.pooling_sizes = d_pooling_layers

        self.pooling_layers = nn.ModuleList()
        self.embedding_layers = nn.ModuleList()

        for pooling_size, hidden_dim, linear_layers, num_heads, num_layers in zip(
            d_pooling_layers,
            d_encoder_hidden_dims,
            d_encoder_linear_layers,
            d_encoder_num_heads,
            d_encoder_num_layers,
        ):
            self.pooling_layers.append(
                GATEncoder(
                    d_features,
                    pooling_size,
                    d_hidden_dim=hidden_dim,
                    d_linear_layers=linear_layers,
                    num_heads=num_heads,
                    num_layers=num_layers,
                    dropout=dropout,
                )
            )
            self.embedding_layers.append(
                GATEncoder(
                    d_features,
                    d_features,
                    d_hidden_dim=hidden_dim,
                    d_linear_layers=linear_layers,
                    num_heads=num_heads,
                    num_layers=num_layers,
                    dropout=dropout,
                )
            )

        self.linear = nn.Sequential(
            nn.Linear(d_features, d_linear),
            nn.ReLU(),
            nn.Linear(d_linear, d_out),
            nn.Dropout(dropout),
        )

    def forward(self, graph_batch: Batch):
        X = graph_batch.x
        A = torch.zeros((X.shape[0], X.shape[0]), device=X.device)
        A[graph_batch.edge_index[0], graph_batch.edge_index[1]] = 1
        batch = graph_batch.batch
        batch_size = torch.max(batch).item() + 1

        # ========= Pooling layers =========
        previous_size = 100
        for i, size in enumerate(self.pooling_sizes):
            S = self.pooling_layers[i](X, A, batch).reshape(batch_size, -1, size)
            S = torch.softmax(
                S,
                dim=-1,
            )
            Z = self.embedding_layers[i](X, A, batch).reshape(
                batch_size, previous_size, -1
            )

            X, A = batch_diffpool(
                S,
                Z,
                extract_blocks(A, previous_size, batch_size),
            )
            previous_size = size

            # Update batch for next pooling layer. Assumes nodes are ordered by graph.
            batch = torch.tensor(
                [k for k in range(batch_size) for _ in range(size)]
            ).to(X.device)

        # If the last pooling layer is 1, then the global mean pooling doesn't do anything
        return self.linear(global_mean_pool(X, batch))


class DiffPool(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_class: int,
        embed_features: List[int],
        num_embed_features: List[int],
        embedding_dim: int = 8,
        dropout: float = 0.1,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        super(DiffPool, self).__init__()

        self.num_features = num_features
        self.num_class = num_class
        self.embed_features = embed_features
        self.num_embed_features = num_embed_features
        self.embedding_dim = embedding_dim
        self.device = device

        self.graph_encoder = DiffPoolEncoder(
            d_features=num_features,
            d_out=num_class,
            d_pooling_layers=[20, 10, 1],
            d_encoder_hidden_dims=[60, 60, 60],
            d_encoder_linear_layers=[[64], [64], [64]],
            d_encoder_num_heads=[3, 3, 3],
            d_encoder_num_layers=[3, 3, 3],
            d_linear=128,
            dropout=dropout,
        )

    def forward(self, x):
        return self.graph_encoder(x)

    def __str__(self):
        return f"DiffPool..."
