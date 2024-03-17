import os

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm

from ..load import load_data


class CFMDataset(Dataset):
    def __init__(
        self,
        split="train",
        process=False,
        cache=True,
        name="CFMDataset",
        test_size=0.1,
        seed=42,
    ):
        """Index is a list of indices of the data to use for this split"""
        self.split = split
        self.index = None
        self.processed_dir = f"./data/processed/{name}/{split}"
        self.cache = cache
        self.test_size = test_size
        self.seed = seed
        self.data = {}
        self.force_process = process

        self.process()

    def should_process(self):
        return self.force_process or not all(
            [
                os.path.exists(os.path.join(self.processed_dir, f"{i}.pt"))
                for i in self.index
            ]
        )

    def process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        X, y, X_test = load_data()
        if self.split == "test":
            X = X_test
            y = None
            self.index = list(range(len(X)))
        elif self.split == "train":
            self.index, _ = train_test_split(
                list(range(len(X))), test_size=self.test_size, random_state=self.seed
            )
        elif self.split == "val":
            _, self.index = train_test_split(
                list(range(len(X))), test_size=self.test_size, random_state=self.seed
            )
        else:
            raise ValueError(f"Unknown split {self.split}")

        if self.should_process():
            for i in tqdm(self.index, desc="Processing data", leave=False):
                if self.split == "train" or self.split == "val":
                    label = torch.tensor(y[i], dtype=torch.long)
                else:
                    label = torch.tensor(-1, dtype=torch.long)
                torch.save(
                    (
                        torch.tensor(X[i], dtype=torch.float32),
                        label,
                    ),
                    os.path.join(self.processed_dir, f"{i}.pt"),
                )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        if self.cache and idx in self.data:
            return self.data[idx]
        else:
            data = torch.load(os.path.join(self.processed_dir, f"{self.index[idx]}.pt"))
            if self.cache:
                self.data[idx] = data
            return data
