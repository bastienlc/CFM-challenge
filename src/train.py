# The MCC loss is adapted from https://github.com/thuml/Versatile-Domain-Adaptation

from typing import Union

import torch
import torch.nn as nn
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Dataset as GeometricDataset
from tqdm import tqdm

from .datasets import CFMDataset
from .loaders import get_test_loader, get_train_loaders
from .utils import TrainLogger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def entropy(input, epsilon=1e-5):
    entropy = -input * torch.log(input + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def mcc_loss(targets, T=2.5):
    batch_size, nb_class = targets.size()
    rescaled_targets = nn.Softmax(dim=1)(targets / T)

    targets_weights = entropy(rescaled_targets).detach()
    targets_weights = 1 + torch.exp(-targets_weights)
    targets_weights = batch_size * targets_weights / torch.sum(targets_weights)

    cov_matrix = (
        rescaled_targets.mul(targets_weights.view(-1, 1))
        .transpose(1, 0)
        .mm(rescaled_targets)
    )
    cov_matrix = cov_matrix / torch.sum(cov_matrix, dim=1)

    return (torch.sum(cov_matrix) - torch.trace(cov_matrix)) / nb_class


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epochs: int = 100,
    batch_size: int = 32,
    load: Union[str, None] = None,
    dataset=CFMDataset,
    device=device,
    test_every: int = 10,
    mu: float = 1.0,
):
    logger = TrainLogger(
        model, optimizer, {"epochs": epochs, "batch_size": batch_size}, load=load
    )

    model, optimizer = logger.load(model, optimizer)

    loss_function = torch.nn.CrossEntropyLoss()

    train_loader, val_loader = get_train_loaders(batch_size=batch_size, dataset=dataset)
    test_loader = get_test_loader(
        batch_size=batch_size, dataset=dataset, shuffle=True, num_workers=0
    )
    num_train_samples = len(train_loader.dataset)
    num_val_samples = len(val_loader.dataset)

    for epoch in range(logger.last_epoch + 1, epochs + 1):
        model.train()
        train_loss = 0
        test_mcc = 0
        train_accuracy = 0
        i = 0

        # TRAIN
        for train_batch in tqdm(train_loader, leave=False):
            i += 1
            # Compute a loss on the test set
            if epoch > 1 and i % test_every == 0:
                test_batch = next(iter(test_loader))
                if issubclass(dataset, TorchDataset):
                    test_output = model(test_batch[0].to(device))
                elif issubclass(dataset, GeometricDataset):
                    test_batch = test_batch.to(device)
                    test_output = model(test_batch)
                else:
                    raise ValueError("Dataset type not recognized")
                mcc = mcc_loss(test_output)
            else:
                mcc = 0

            if issubclass(dataset, TorchDataset):
                target = train_batch[1].to(device)
                train_output = model(train_batch[0].to(device))
            elif issubclass(dataset, GeometricDataset):
                train_batch = train_batch.to(device)
                target = train_batch.y
                train_output = model(train_batch)
            else:
                raise ValueError("Dataset type not recognized")

            loss = loss_function(train_output, target) + mu * mcc
            train_loss += loss.item()
            test_mcc += mcc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            prediction = torch.argmax(train_output, dim=1)
            train_accuracy += (prediction == target).sum().item()

        scheduler.step()
        logger.log(
            epoch,
            train_loss=train_loss / num_train_samples,
            additional_metrics={
                "train_accuracy": train_accuracy / num_train_samples,
                "test_mcc": test_mcc,
            },
        )

        # EVAL
        model.eval()
        with torch.no_grad():
            val_loss = 0
            accuracy = 0
            for batch in val_loader:
                if issubclass(dataset, TorchDataset):
                    target = batch[1].to(device)
                    output = model(batch[0].to(device))
                elif issubclass(dataset, GeometricDataset):
                    batch = batch.to(device)
                    target = batch.y
                    output = model(batch)
                else:
                    raise ValueError("Dataset type not recognized")
                val_loss += loss_function(output, target).item()
                accuracy += (torch.argmax(output, dim=1) == target).sum().item()

        logger.log(
            epoch,
            val_loss=val_loss / num_val_samples,
            val_accuracy=accuracy / num_val_samples,
            additional_metrics={"learning_rate": optimizer.param_groups[0]["lr"]},
        )
        logger.save(model, optimizer, val_accuracy=accuracy / num_val_samples)
        logger.print(epoch)

    return logger.save_dir
