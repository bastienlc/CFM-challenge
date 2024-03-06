"""The main way to load the data is through the load_data function which reads the CSV file, and is cached using the numpy_cache decorator. However this is always quite slow so the CFMDataset dataset preprocesses the data and stores it as .pt files."""

import os

import numpy as np
import pandas as pd
import torch
from data_cache import numpy_cache
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


@numpy_cache
def load_data(dummy=False, shuffle=True, seed=42):
    """features : {0: "venue", 1: "order_id", 2: "action", 3: "side", 4: "price", 5: "bid", 6: "ask", 7: "bid_size", 9: "ask_size", 9: "trade", 10: "flux"}"""
    np.random.seed(seed)
    # LOAD
    if dummy:
        X = pd.read_csv("data/small_x_train.csv")
        y = pd.read_csv("data/small_y_train.csv")
        X_test = pd.read_csv("data/small_x_train.csv")
    else:
        X = pd.read_csv("data/x_train.csv")
        y = pd.read_csv("data/y_train.csv")
        X_test = pd.read_csv("data/x_test.csv")

    # LABELS
    labels_to_encode = ["action", "side", "trade"]
    for label in labels_to_encode:
        label_encoder = LabelEncoder()
        X[label] = label_encoder.fit_transform(X[label])
        X_test[label] = label_encoder.transform(X_test[label])

    # FILL
    fill_values = X.mean()
    nb = X.isna().sum().sum()
    if nb > 0:
        print("Filling", nb, "NAs in train")
    X = X.fillna(fill_values)

    nb = X_test.isna().sum().sum()
    if nb > 0:
        print("Filling", nb, "NAs in test")
    X_test = X_test.fillna(fill_values)

    # NUMPY AND RESHAPE
    X = X.to_numpy()
    X = X.reshape(-1, 100, X.shape[1])
    X = X[:, :, 1:]
    X_test = X_test.to_numpy()
    X_test = X_test.reshape(-1, 100, X_test.shape[1])
    X_test = X_test[:, :, 1:]

    # Y
    y = y.set_index("obs_id").to_numpy().reshape(-1)

    # SHUFFLE
    if shuffle:
        shuffled_index = np.random.permutation(X.shape[0])
    else:
        shuffled_index = np.arange(X.shape[0])

    return X[shuffled_index, :, :], y[shuffled_index], X_test


class CFMDataset(Dataset):
    def __init__(
        self,
        index,
        split="train",
        process=False,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        """Index is a list of indices of the data to use for this split"""
        self.split = split
        self.index = index
        self.processed_dir = f"./data/processed/{split}"
        self.device = device

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
        if self.split == "train" or self.split == "val":
            for i in tqdm(self.index, desc="Processing data", leave=False):
                torch.save(
                    (
                        torch.tensor(X[i], dtype=torch.float32, device=self.device),
                        torch.tensor(y[i], dtype=torch.long, device=self.device),
                    ),
                    os.path.join(self.processed_dir, f"{i}.pt"),
                )
        elif self.split == "test":
            for i in tqdm(self.index, desc="Processing data", leave=False):
                torch.save(
                    (
                        torch.tensor(
                            X_test[i], dtype=torch.float32, device=self.device
                        ),
                        torch.tensor(-1, dtype=torch.long, device=self.device),
                    ),
                    os.path.join(self.processed_dir, f"{i}.pt"),
                )
        else:
            raise ValueError("split should be 'train', 'val' or 'test'")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        return torch.load(os.path.join(self.processed_dir, f"{self.index[idx]}.pt"))


def get_train_loaders(batch_size=32, shuffle=True, test_size=0.1, seed=42):
    num_training_samples = 160800

    train_index, val_index = train_test_split(
        range(num_training_samples), test_size=test_size, random_state=seed
    )

    train_dataset = CFMDataset(train_index, split="train")
    val_dataset = CFMDataset(val_index, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    return train_loader, val_loader


def get_test_loader(batch_size=32):
    num_test_samples = 81600
    test_index = range(num_test_samples)
    test_dataset = CFMDataset(test_index, split="test")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader
