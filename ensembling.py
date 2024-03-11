import os

import numpy as np
import torch
from sklearn.linear_model import Ridge

from src.datasets import CFMGraphDataset
from src.loaders import get_test_loader, get_train_loaders
from src.models import GATEncoder
from src.utils import predict, save

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_ensemble(model, loader, k: int, split: str):
    predictions = []
    files = [k + 1 for k in range(26)]

    for k in files:

        if os.path.exists(f"runs/ensemble/{k}/probas_{split}.pt"):
            probas = torch.load(f"runs/ensemble/{k}/probas_{split}.pt")
        else:
            print(f"Computing probas with model {k}")
            model.load_state_dict(torch.load(f"runs/ensemble/{k}/model.pt"))

            _, _, probas = predict(
                model,
                loader,
                device,
                is_torch_geometric,
            )

            torch.save(probas, f"runs/ensemble/{k}/probas_{split}.pt")

        predictions.append(probas)

    return np.stack(predictions, axis=0)  # (n_models, n_samples, n_classes)


if __name__ == "__main__":
    train_loader, val_loader = get_train_loaders(
        batch_size=256, shuffle=False, dataset=CFMGraphDataset
    )
    test_loader = get_test_loader(batch_size=256, dataset=CFMGraphDataset)
    is_torch_geometric = True

    model = GATEncoder(
        d_features=7,
        d_edges=5,
        d_out=24,
        d_hidden_dim=300,
        num_layers=3,
        num_heads=3,
        d_linear_layers=[256],
        dropout=0.01,
        activation="ReLU",
    ).to(device)

    test_predictions = predict_ensemble(model, test_loader, 26, "test")
    train_predictions = predict_ensemble(model, train_loader, 26, "train")
    val_predictions = predict_ensemble(model, val_loader, 26, "val")

    train_y_true = np.concatenate([batch.y.cpu().numpy() for batch in train_loader])
    val_y_true = np.concatenate([batch.y.cpu().numpy() for batch in val_loader])

    # replace nas by 0
    train_predictions = np.nan_to_num(train_predictions)
    val_predictions = np.nan_to_num(val_predictions)
    test_predictions = np.nan_to_num(test_predictions)

    stacking_models = [Ridge(alpha=0.1, tol=0.01) for _ in range(24)]

    for i in range(24):
        print(f"Training model {i}")
        stacking_models[i].fit(train_predictions[:, :, i].T, train_y_true == i)

    train_predictions = np.stack(
        [
            model.predict(train_predictions[:, :, i].T)
            for i, model in enumerate(stacking_models)
        ],
        axis=1,
    )

    val_predictions = np.stack(
        [
            model.predict(val_predictions[:, :, i].T)
            for i, model in enumerate(stacking_models)
        ],
        axis=1,
    )

    test_predictions = np.stack(
        [
            model.predict(test_predictions[:, :, i].T)
            for i, model in enumerate(stacking_models)
        ],
        axis=1,
    )

    # save to file
    np.save("runs/ensemble/train_predictions.npy", train_predictions)
    np.save("runs/ensemble/val_predictions.npy", val_predictions)
    np.save("runs/ensemble/test_predictions.npy", test_predictions)

    # predict
    y_pred = test_predictions.argmax(axis=1)
    save(y_pred, "solution.csv")

    print("Validation accuracy:", (val_predictions.argmax(axis=1) == val_y_true).mean())
    print("Train accuracy:", (train_predictions.argmax(axis=1) == train_y_true).mean())
