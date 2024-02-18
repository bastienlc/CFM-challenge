import torch
import torch.nn as nn


class Base(nn.Module):
    def __init__(
        self,
        num_features,
        num_class,
        embed_features,
        num_embed_features,
        encode_features,
        embedding_dim=8,
        d_hidden=128,
        num_layers=2,
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

        self.embbeddings = nn.ParameterList(
            [nn.Parameter(torch.randn(n, embedding_dim)) for n in num_embed_features]
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

    def forward(self, x):
        input = torch.zeros(x.shape[0], x.shape[1], self.input_dim).to(x.device)
        count = 0
        for i in self.encode_features:
            input[:, :, count] = self.encoder(x[:, :, i])
            count += 1

        for i, j in enumerate(self.embed_features):
            input[:, :, count : count + self.embedding_dim] = self.embbeddings[i][
                x[:, :, j].type(torch.int)
            ]
            count += self.embedding_dim

        for i in range(self.num_features):
            if i not in self.embed_features and i not in self.encode_features:
                input[:, :, count] = x[:, :, i]
                count += 1

        output, (h, c) = self.lstm(input)

        return self.linear(
            h.transpose(0, 1).reshape(-1, 2 * self.num_layers * self.d_hidden)
        ).softmax(dim=1)

    def __str__(self):
        return f"Base(num_features={self.num_features}, num_class={self.num_class}, embed_features={self.embed_features}, num_embed_features={self.num_embed_features}, encode_features={self.encode_features}, embedding_dim={self.embedding_dim}, d_hidden={self.d_hidden}, num_layers={self.num_layers})"
