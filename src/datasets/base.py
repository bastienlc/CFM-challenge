import os

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from ..load import load_data


class CFMDataset(Dataset):
    def __init__(
        self,
        index,
        split="train",
        process=False,
        cache=True,
    ):
        """Index is a list of indices of the data to use for this split"""
        self.split = split
        self.index = index
        self.processed_dir = f"./data/processed/{split}"
        self.cache = cache
        self.data = {}

        if process or self.should_process():
            self.process()

    def should_process(self):
        return not all(
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

        for i in tqdm(self.index, desc="Processing data", leave=False):
            if self.split == "train" or self.split == "val":
                label = torch.tensor(y[i], dtype=torch.long)
            else:
                label = torch.tensor(-1, dtype=torch.long)
            torch.save(
                (
                    torch.tensor(X[i], dtype=torch.float32),
                    torch.tensor(label, dtype=torch.long),
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
