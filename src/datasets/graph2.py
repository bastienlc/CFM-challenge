import os

import networkx as nx
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, Dataset
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from tqdm import tqdm
from collections import Counter

from ..load import load_data


class CFMGraphDataset(Dataset):
    def __init__(
        self,
        split="train",
        process=False,
        cache=True,
        randomize=False,
        name="CFMGraphDataset",
        test_size=0.1,
        seed=42,
    ):
        """Index is a list of indices of the data to use for this split"""
        self.split = split
        self.index = None
        self.cache = cache
        self.randomize = randomize
        self.test_size = test_size
        self.seed = seed
        self.data = {}
        self.name = name
        self.force_process = process

        self._processed_dir = os.path.join("data", "processed", self.name, self.split)
        self.process()
        self._processed_file_names = [f"{i}.pt" for i in self.index]

    @property
    def processed_dir(self):
        return self._processed_dir

    @property
    def processed_file_names(self):
        return self._processed_file_names

    def should_process(self):
        return self.force_process or not all(
            [
                os.path.exists(os.path.join(self.processed_dir, f"{i}.pt"))
                for i in self.index
            ]
        )

    def process(self):
        """
        node features : price (4), bid (5), ask (6), bid_size (7), ask_size (8), flux (10)
        edge features : action (2), side (3), trade (9)
        """
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

        X, X_obs, y, X_test, X_test_obs, features = load_data()
        if self.split == "test":
            X = X_test
            y = None
            self.index = list(set(X_test_obs))
            counter = dict(Counter(X_test_obs))
        elif self.split == "train":
            self.index, _ = train_test_split(
                list(set(X_obs)), test_size=self.test_size, random_state=self.seed
            )
            counter = dict(Counter(X_obs))
        elif self.split == "val":
            _, self.index = train_test_split(
                list(set(X_obs)), test_size=self.test_size, random_state=self.seed
            )
            counter = dict(Counter(X_obs))
        else:
            raise ValueError(f"Unknown split {self.split}")

        counter = {k: counter[k] for k in self.index}

        current_index = 0
        print("\nStarting processing...")
        if self.should_process():
            for i in tqdm(self.index, desc="Processing data", leave=False):
                if self.split == "train" or self.split == "val":
                    label = torch.tensor(y[i], dtype=torch.long)
                else:
                    label = torch.tensor(-1, dtype=torch.long)
                graph = nx.Graph()
                node_features = np.array(["bid_ask_spread", "side", "price", "bid_size", "ask_size", "venue", "flux"])
                node_features_indices = [np.where(features == node_feature)[0][0] for node_feature in node_features]

                edge_temp_features = np.array(["trade", "Limit Order"])
                edge_temp_features_indices = [np.where(features == edge_feature)[0][0] for edge_feature in edge_temp_features]

                edge_id_features = np.array(["trade", "Limit Order"])
                edge_id_features_indices = [np.where(features == edge_feature)[0][0] for edge_feature in edge_id_features]

                curr_X = X[current_index : current_index + counter[i]]
                current_index += counter[i]
                node_arr = torch.zeros((len(curr_X), len(node_features)), dtype=torch.float32)
                order_ids_keys = {}
                for k, row in enumerate(curr_X):
                    order_id = row[1]

                    graph.add_node(k)
                    node_arr[k, :] = torch.tensor(
                        [row[j] for j in node_features_indices],
                        dtype=torch.float32,
                    )
                    temp_attr = torch.tensor(
                        [row[j] for j in edge_temp_features_indices],
                        dtype=torch.int,
                    )
                    graph.add_edge(k, k-1, edge_attr=temp_attr) # Temporal edge

                    id_attr = torch.tensor(
                        [row[j] for j in edge_id_features_indices],
                        dtype=torch.int,
                    )

                    if order_id in order_ids_keys:
                        graph.add_edge(k, order_ids_keys[order_id][-1], edge_attr=id_attr)
                        order_ids_keys[order_id].append(k)
                    else:
                        order_ids_keys[order_id] = [k]
                try :
                    edge_index, _ = from_scipy_sparse_matrix(nx.adjacency_matrix(graph))
                except :
                    print("Index fucked up : ", i)

                torch.save(
                    Data(
                        edge_index=torch.tensor(edge_index),
                        x=torch.tensor(node_features),
                        edge_attr=torch.stack(
                            list(nx.get_edge_attributes(graph, "edge_attr").values())
                        ).repeat_interleave(2, dim=0),
                        y=label,
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

            if self.randomize and self.split == "train":
                random_part = 0.1 * torch.randn(data.x.size(0), data.x.size(1))
                random_part[:, :2] = 0
                return Data(
                    x=data.x + random_part,
                    edge_index=data.edge_index,
                    edge_attr=data.edge_attr,
                    y=data.y,
                )
            else:
                return data

    def len(self):
        return self.__len__()

    def get(self, idx):
        return self.__getitem__(idx)

    def get_y(self):
        return np.concatenate([self[i].y.numpy() for i in range(len(self))], axis=0)
