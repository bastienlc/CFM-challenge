import os

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Dataset as GeometricDataset

from src.datasets import CFMDataset, CFMGraphDataset, FeaturesDataset
from src.loaders import get_test_loader, get_train_loaders
from src.models import (
    Base,
    GATEncoder,
    GENEncoder,
    GeneralEncoder,
    PDNEncoder,
    PNAEncoder,
    TransformerEncoder,
)
from src.utils import predict, save

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_model_split(model, path, dataset, split):
    if split == "test":
        loader = get_test_loader(batch_size=1024, shuffle=False, dataset=dataset)
    elif split == "train":
        loader, _ = get_train_loaders(batch_size=1024, shuffle=False, dataset=dataset)
    elif split == "val":
        _, loader = get_train_loaders(batch_size=1024, shuffle=False, dataset=dataset)

    if os.path.exists(os.path.join(path, f"probas_{split}.pt")):
        probas = torch.load(os.path.join(path, f"probas_{split}.pt"))
    else:
        model = model.to(device)
        model.load_state_dict(torch.load(os.path.join(path, "model.pt")))
        model.eval()

        _, _, probas = predict(
            model,
            loader,
            device,
            issubclass(dataset, GeometricDataset),
        )

        torch.save(probas, os.path.join(path, f"probas_{split}.pt"))

    return probas


def predict_ensemble(models_list):
    train_probas = []
    val_probas = []
    test_probas = []

    for k, (model, path, dataset) in enumerate(models_list):
        print(f"Predicting model {path}... ({k + 1}/{len(models_list)})")
        train_probas.append(predict_model_split(model, path, dataset, "train"))
        val_probas.append(predict_model_split(model, path, dataset, "val"))
        test_probas.append(predict_model_split(model, path, dataset, "test"))
    return (
        np.stack(train_probas, axis=0),
        np.stack(val_probas, axis=0),
        np.stack(test_probas, axis=0),
    )


def aggregate_probas(probas):
    return probas.mean(axis=0).argmax(axis=1)


if __name__ == "__main__":

    models_list = [
        (
            Base(
                num_features=11,
                num_class=24,
                embed_features=[0, 2, 3, 9],
                num_embed_features=[6, 3, 2, 2],
                encode_features=[],
                d_hidden=128,
                embedding_dim=8,
                num_layers=2,
                dropout=0.01,
            ),
            "runs/lstm",
            CFMDataset,
        ),
        (
            GATEncoder(
                d_features=7,
                d_edges=5,
                d_out=24,
                d_hidden_dim=300,
            ),
            "runs/mcc_2",
            CFMGraphDataset,
        ),
        (
            GATEncoder(
                d_features=7,
                d_edges=5,
                d_out=24,
                d_hidden_dim=300,
            ),
            "runs/ensemble/18",
            CFMGraphDataset,
        ),
        (
            GATEncoder(
                d_features=7,
                d_edges=5,
                d_out=24,
                d_hidden_dim=300,
            ),
            "runs/ensemble/9",
            CFMGraphDataset,
        ),
        (
            GATEncoder(
                d_features=7,
                d_edges=5,
                d_out=24,
                d_hidden_dim=300,
            ),
            "runs/ensemble/8",
            CFMGraphDataset,
        ),
        (
            GATEncoder(
                d_features=7,
                d_edges=5,
                d_out=24,
                d_hidden_dim=300,
            ),
            "runs/ensemble/2",
            CFMGraphDataset,
        ),
        (
            GATEncoder(
                d_features=7,
                d_edges=5,
                d_out=24,
                d_hidden_dim=300,
            ),
            "runs/night_test",
            CFMGraphDataset,
        ),
        (
            GATEncoder(
                d_features=7,
                d_edges=5,
                d_out=24,
                d_hidden_dim=300,
            ),
            "runs/ensemble/6",
            CFMGraphDataset,
        ),
        (
            GATEncoder(
                d_features=7,
                d_edges=5,
                d_out=24,
                d_hidden_dim=300,
            ),
            "runs/ensemble/22",
            CFMGraphDataset,
        ),
        (
            GATEncoder(
                d_features=7,
                d_edges=5,
                d_out=24,
                d_hidden_dim=300,
            ),
            "runs/ensemble/16",
            CFMGraphDataset,
        ),
        (
            GATEncoder(
                d_features=7,
                d_edges=5,
                d_out=24,
                d_hidden_dim=300,
            ),
            "runs/ensemble/1",
            CFMGraphDataset,
        ),
        (
            GATEncoder(
                d_features=7,
                d_edges=5,
                d_out=24,
                d_hidden_dim=300,
            ),
            "runs/ensemble/13",
            CFMGraphDataset,
        ),
        (
            GATEncoder(
                d_features=7,
                d_edges=5,
                d_out=24,
                d_hidden_dim=300,
            ),
            "runs/ensemble/4",
            CFMGraphDataset,
        ),
        (
            PNAEncoder(
                d_features=7,
                d_edges=5,
                d_out=24,
                d_hidden_dim=300,
            ),
            "runs/pna",
            CFMGraphDataset,
        ),
        (
            TransformerEncoder(
                d_features=7,
                d_edges=5,
                d_out=24,
                d_hidden_dim=300,
            ),
            "runs/transformer",
            CFMGraphDataset,
        ),
        (
            GeneralEncoder(
                d_features=7,
                d_edges=5,
                d_out=24,
                d_hidden_dim=300,
            ),
            "runs/general",
            CFMGraphDataset,
        ),
        (
            PDNEncoder(
                d_features=7,
                d_edges=5,
                d_out=24,
                d_hidden_dim=300,
            ),
            "runs/pdn",
            CFMGraphDataset,
        ),
        (
            GENEncoder(
                d_features=7,
                d_edges=5,
                d_out=24,
                d_hidden_dim=300,
            ),
            "runs/gen",
            CFMGraphDataset,
        ),
        (
            nn.Sequential(
                nn.Linear(84, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 24),
            ),
            "runs/mlp",
            FeaturesDataset,
        ),
        (
            GATEncoder(
                d_features=7,
                d_edges=5,
                d_out=24,
                d_hidden_dim=1000,
                num_layers=3,
                num_heads=8,
                d_linear_layers=[512, 256],
                dropout=0.1,
                activation="ReLU",
            ),
            "runs/big_1",
            CFMGraphDataset,
        ),
        (
            GATEncoder(
                d_features=7,
                d_edges=5,
                d_out=24,
                d_hidden_dim=1000,
                num_layers=3,
                num_heads=8,
                d_linear_layers=[512, 256],
                dropout=0.1,
                activation="ReLU",
            ),
            "runs/big_2",
            CFMGraphDataset,
        ),
    ]

    train_probas, val_probas, test_probas = predict_ensemble(models_list)

    train_predictions = aggregate_probas(train_probas)
    val_predictions = aggregate_probas(val_probas)
    test_predictions = aggregate_probas(test_probas)

    np.save("runs/train_predictions.npy", train_predictions)
    np.save("runs/val_predictions.npy", val_predictions)
    np.save("runs/test_predictions.npy", test_predictions)

    save(test_predictions, "solution.csv")

    train_loader, val_loader = get_train_loaders(
        batch_size=1024, shuffle=False, dataset=CFMGraphDataset
    )
    train_labels = np.concatenate([batch.y.cpu().numpy() for batch in train_loader])
    val_labels = np.concatenate([batch.y.cpu().numpy() for batch in val_loader])

    print("Train accuracy:", (train_predictions == train_labels).mean())
    print("Validation accuracy:", (val_predictions == val_labels).mean())
