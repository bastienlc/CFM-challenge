import os

import networkx as nx
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from tqdm import tqdm
import numpy as np

from ..load import load_data


class CFMGraphDataset(Dataset):
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
        self.cache = cache
        self.data = {}

        super(CFMGraphDataset, self).__init__("./data")

        if process or self.should_process():
            self.process()

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, "processed", self.split)

    @property
    def processed_file_names(self):
        return [f"{i}.pt" for i in self.index]

    def should_process(self):
        return not all(
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

            curr_X = X[X_obs == i]
            node_arr = torch.zeros((len(curr_X), len(node_features)), dtype=torch.float32)
            order_ids_keys = {}
            for k, row in enumerate(curr_X):
                order_id = row[1]
                venue = row[0]

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

            edge_index, _ = from_scipy_sparse_matrix(nx.adjacency_matrix(graph))

            torch.save(
                Data(
                    edge_index=edge_index,
                    x=node_features,
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
            return data

    def len(self):
        return self.__len__()

    def get(self, idx):
        return self.__getitem__(idx)
