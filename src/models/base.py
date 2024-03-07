from typing import List

import numpy as np
import torch
import torch.nn as nn


class Base(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_class: int,
        embed_features: List[int],
        num_embed_features: List[int],
        encode_features: List[int],
        embedding_dim: int = 8,
        d_hidden: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        super(Base, self).__init__()

        # Model parameters
        self.num_features = num_features
        self.num_class = num_class
        self.embed_features = embed_features
        self.num_embed_features = num_embed_features
        self.encode_features = encode_features
        self.embedding_dim = embedding_dim
        self.d_hidden = d_hidden
        self.num_layers = num_layers
        self.device = device

        self.features_not_modified = [
            i
            for i in range(self.num_features)
            if i not in self.embed_features and i not in self.encode_features
        ]

        self.embbeddings = nn.Embedding(
            sum(self.num_embed_features),
            self.embedding_dim,
        )
        self.embeddings_offset = torch.tensor(
            np.concatenate(([0], np.cumsum(self.num_embed_features)[:-1])),
            dtype=torch.int,
            device=self.device,
        )

        self.input_dim = sum(
            [
                self.embedding_dim if i in self.embed_features else 1
                for i in range(self.num_features)
            ]
        )

        self.lstm = nn.LSTM(
            self.input_dim,
            self.d_hidden,
            self.num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.linear = nn.Sequential(
            nn.Linear(2 * self.num_layers * self.d_hidden, self.d_hidden),
            nn.ReLU(),
            nn.Linear(self.d_hidden, self.d_hidden // 2),
            nn.ReLU(),
            nn.Linear(self.d_hidden // 2, self.num_class),
        )

        self.encoder = lambda u: torch.sin(torch.pi * u / 100)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size = x.shape[0]
        input = torch.zeros(batch_size, x.shape[1], self.input_dim, device=x.device)
        count = 0

        # Encoded features
        input[:, :, count : count + len(self.encode_features)] = self.encoder(
            x[:, :, self.encode_features]
        )
        count += len(self.encode_features)

        # Embedded features
        input[:, :, count : count + self.embedding_dim * len(self.embed_features)] = (
            self.embbeddings(
                x[:, :, self.embed_features].type(torch.int) + self.embeddings_offset
            )
        ).reshape(batch_size, 100, self.embedding_dim * len(self.embed_features))
        count += self.embedding_dim * len(self.embed_features)

        # Other features
        input[:, :, count:] = x[
            :,
            :,
            self.features_not_modified,
        ]

        output, (h, c) = self.lstm(input)

        return self.linear(
            self.dropout(
                h.transpose(0, 1).reshape(-1, 2 * self.num_layers * self.d_hidden)
            )
        ).softmax(dim=1)

    def __str__(self):
        return f"Base(num_features={self.num_features}, num_class={self.num_class}, embed_features={self.embed_features}, num_embed_features={self.num_embed_features}, encode_features={self.encode_features}, embedding_dim={self.embedding_dim}, d_hidden={self.d_hidden}, num_layers={self.num_layers}, dropout={self.dropout.p})"
