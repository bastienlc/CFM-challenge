import os

import networkx as nx
import torch
from torch_geometric.data import Data, Dataset
from tqdm import tqdm

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

            graph = nx.Graph()
            features = torch.zeros((len(X[i]), 9), dtype=torch.float32)
            venues_previous_keys = {}
            order_ids_keys = {}
            for k, row in enumerate(X[i]):
                order_id = row[1]
                venue = row[0]

                graph.add_node(k)
                features[k, :] = torch.tensor(row[2:])

                if venue in venues_previous_keys:
                    graph.add_edge(k, venues_previous_keys[venue])
                    graph.add_edge(venues_previous_keys[venue], k)
                venues_previous_keys[venue] = k

                if order_id in order_ids_keys:
                    graph.add_edge(k, order_ids_keys[order_id][-1])
                    graph.add_edge(order_ids_keys[order_id][-1], k)
                    order_ids_keys[order_id].append(k)
                else:
                    order_ids_keys[order_id] = [k]

            adj = torch.tensor(
                nx.adjacency_matrix(graph).todense(),
                dtype=torch.float32,
            )
            edge_index = adj.nonzero().t().contiguous()
            torch.save(
                Data(x=features, y=label, edge_index=edge_index),
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
