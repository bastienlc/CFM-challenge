import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class FileDataset(Dataset):
    def __init__(
        self,
        split="train",
        test_size=0.1,
        seed=42,
    ):
        """Index is a list of indices of the data to use for this split"""
        self.split = split
        self.test_size = test_size
        self.seed = seed
        self.data = {}

        if self.split == "train" or self.split == "val":
            self.X = pd.read_parquet(f"data/features_train.parquet").to_numpy()[:, 1:]
            self.y = pd.read_csv("data/y_train.csv").to_numpy()[:, 1]
        elif self.split == "test":
            self.X = pd.read_parquet(f"data/features_test.parquet").to_numpy()[:, 1:]
            self.y = np.full(len(self.X), -1, dtype=float)

        if self.split == "test":
            self.index = list(range(len(self.X)))
        elif self.split == "train":
            self.index, _ = train_test_split(
                list(range(len(self.X))),
                test_size=self.test_size,
                random_state=self.seed,
            )
        elif self.split == "val":
            _, self.index = train_test_split(
                list(range(len(self.X))),
                test_size=self.test_size,
                random_state=self.seed,
            )
        else:
            raise ValueError(f"Unknown split {self.split}")

        self.X = self.X[self.index]
        self.y = self.y[self.index]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
