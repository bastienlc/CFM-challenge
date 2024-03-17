import os

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.base import ClassifierMixin
from torch_geometric.data import Dataset as GeometricDataset

from src.datasets import CFMDataset, CFMGraphDataset, FeaturesDataset, FileDataset
from src.loaders import get_test_loader, get_train_loaders
from src.models import (
    Base,
    GATEncoder,
    GENEncoder,
    GeneralEncoder,
    PDNEncoder,
    PNAEncoder,
    ResidualModel,
    TransformerEncoder,
)
from src.utils import predict, save

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model, path):
    if isinstance(model, nn.Module):
        model = model.to(device)
        model.load_state_dict(torch.load(os.path.join(path, "model.pt")))
        model.eval()
    elif isinstance(model, ClassifierMixin):
        model.load(path)
    else:
        raise ValueError("Model type not recognized")

    return model


def predict_model_split(model, path, dataset, split):
    if os.path.exists(os.path.join(path, f"probas_{split}.pt")):
        probas = torch.load(os.path.join(path, f"probas_{split}.pt"))
    else:
        if isinstance(model, nn.Module):
            model = load_model(model, path)

            if split == "test":
                loader = get_test_loader(
                    batch_size=1024, shuffle=False, dataset=dataset
                )
            elif split == "train":
                loader, _ = get_train_loaders(
                    batch_size=1024, shuffle=False, dataset=dataset
                )
            elif split == "val":
                _, loader = get_train_loaders(
                    batch_size=1024, shuffle=False, dataset=dataset
                )

            _, _, probas = predict(
                model,
                loader,
                device,
                issubclass(dataset, GeometricDataset),
            )
        else:
            X = dataset(split=split).X
            model = load_model(model, path)
            probas = model.predict_proba(X)

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
    return train_probas, val_probas, test_probas


def accuracy(split, path, dataset):
    probas = torch.load(os.path.join(path, f"probas_{split}.pt"))
    predictions = probas.argmax(axis=1)

    if isinstance(dataset, FileDataset):
        labels = dataset(split=split).y
    else:
        if split == "train":
            loader, _ = get_train_loaders(
                batch_size=1024, shuffle=False, dataset=dataset
            )
        elif split == "val":
            _, loader = get_train_loaders(
                batch_size=1024, shuffle=False, dataset=dataset
            )
        elif split == "test":
            loader = get_test_loader(batch_size=1024, shuffle=False, dataset=dataset)

        if issubclass(dataset, GeometricDataset):
            labels = np.concatenate([batch.y.cpu().numpy() for batch in loader])
        else:
            labels = np.concatenate([batch[1].cpu().numpy() for batch in loader])

    return (predictions == labels).mean()


def aggregate_probas(probas):
    return probas.mean(axis=0).argmax(axis=1)


if __name__ == "__main__":

    models_list = [
        (ResidualModel(), "runs/residuals", FileDataset),
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

    weights = [
        1,
        0.8,
        0.2,
        0.2,
        0.2,
        0.2,
        0.2,
        0.2,
        0.2,
        0.2,
        0.2,
        0.2,
        0.2,
        0.2,
        0.9,
        0.9,
        0.9,
        0.9,
        0.9,
        0.9,
        0.5,
        0.5,
    ]
    weights = np.array(weights) / np.sum(weights)

    test_predictions = aggregate_probas(
        weights[:, None, None] * np.stack(test_probas, axis=0)
    )
    # We don't make predictions on train and val because they may have different sizes

    save(test_predictions, "solution.csv")

    if False:
        train_accuracies = []
        val_accuracies = []
        for k, (_, path, dataset) in enumerate(models_list):
            print(f"Computing accuracies... ({k + 1}/{len(models_list)})")
            train_accuracies.append(accuracy("train", path, dataset))
            val_accuracies.append(accuracy("val", path, dataset))

        torch.save(train_accuracies, "runs/train_accuracies.pt")
        torch.save(val_accuracies, "runs/val_accuracies.pt")
    else:
        train_accuracies = torch.load("runs/train_accuracies.pt")
        val_accuracies = torch.load("runs/val_accuracies.pt")

    plt.figure()
    plt.bar(
        np.arange(len(models_list)) - 0.1, train_accuracies, label="Train", width=0.2
    )
    plt.bar(np.arange(len(models_list)) + 0.1, val_accuracies, label="Val", width=0.2)
    plt.plot(np.arange(len(models_list)), weights, label="Weights", color="g")
    plt.ylim(0, 1.1)

    plt.xticks(
        range(len(models_list)), [path for _, path, _ in models_list], rotation=90
    )
    plt.legend()
    plt.show()
