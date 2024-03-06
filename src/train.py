from typing import Union

import numpy as np
import torch
import torch.nn as nn

from .load import get_data_loaders
from .utils import TrainLogger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
    device: torch.device = device,
    load: Union[str, None] = None,
):
    logger = TrainLogger(
        model, optimizer, {"epochs": epochs, "batch_size": batch_size}, load=load
    )

    model, optimizer = logger.load(model, optimizer)

    loss_function = torch.nn.CrossEntropyLoss()

    train_loader, val_loader = get_data_loaders(
        X_train, y_train, X_val, y_val, device, batch_size=batch_size
    )

    for epoch in range(logger.last_epoch + 1, epochs + 1):
        model.train()
        train_loss = 0
        train_accuracy = 0

        # TRAIN
        for data, target in tqdm(train_loader, leave=False):
            output = model(data)
            loss = loss_function(output, target)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            prediction = torch.argmax(output, dim=1)
            train_accuracy += (prediction == target).sum().item()

        scheduler.step()
        logger.log(
            epoch,
            train_loss=train_loss / len(X_train),
            additional_metrics={"train_accuracy": train_accuracy / len(X_train)},
        )

        # EVAL
        model.eval()
        with torch.no_grad():
            val_loss = 0
            accuracy = 0
            for data, target in val_loader:
                output = model(data)
                val_loss += loss_function(output, target).item()
                accuracy += (torch.argmax(output, dim=1) == target).sum().item()

        logger.log(
            epoch,
            val_loss=val_loss / len(X_val),
            val_accuracy=accuracy / len(X_val),
            additional_metrics={"learning_rate": optimizer.param_groups[0]["lr"]},
        )
        logger.save(model, optimizer, val_accuracy=accuracy / len(X_val))
        logger.print(epoch)

    return model
