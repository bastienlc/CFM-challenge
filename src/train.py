from typing import Union

import torch
import torch.nn as nn
from tqdm import tqdm

from .datasets import CFMGraphDataset
from .loaders import get_train_loaders
from .utils import TrainLogger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epochs: int = 100,
    batch_size: int = 32,
    load: Union[str, None] = None,
):
    logger = TrainLogger(
        model, optimizer, {"epochs": epochs, "batch_size": batch_size}, load=load
    )

    model, optimizer = logger.load(model, optimizer)

    loss_function = torch.nn.CrossEntropyLoss()

    train_loader, val_loader = get_train_loaders(
        batch_size=batch_size, dataset=CFMGraphDataset
    )
    num_train_samples = len(train_loader.dataset)
    num_val_samples = len(val_loader.dataset)

    for epoch in range(logger.last_epoch + 1, epochs + 1):
        model.train()
        train_loss = 0
        train_accuracy = 0

        # TRAIN
        for batch in tqdm(train_loader, leave=False):
            output = model(batch)
            loss = loss_function(output, batch.y)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            prediction = torch.argmax(output, dim=1)
            train_accuracy += (prediction == batch.y).sum().item()

        scheduler.step()
        logger.log(
            epoch,
            train_loss=train_loss / num_train_samples,
            additional_metrics={"train_accuracy": train_accuracy / num_train_samples},
        )

        # EVAL
        model.eval()
        with torch.no_grad():
            val_loss = 0
            accuracy = 0
            for batch in val_loader:
                output = model(batch)
                val_loss += loss_function(output, batch.y).item()
                accuracy += (torch.argmax(output, dim=1) == batch.y).sum().item()

        logger.log(
            epoch,
            val_loss=val_loss / num_val_samples,
            val_accuracy=accuracy / num_val_samples,
            additional_metrics={"learning_rate": optimizer.param_groups[0]["lr"]},
        )
        logger.save(model, optimizer, val_accuracy=accuracy / num_val_samples)
        logger.print(epoch)

    return model
