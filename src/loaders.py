from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader as GeometricDataLoader

from .datasets import CFMDataset


def get_train_loaders(
    batch_size=32,
    shuffle=True,
    test_size=0.1,
    seed=42,
    dataset=CFMDataset,
    num_workers=4,
):
    num_training_samples = 154992  # after removing outliers

    train_index, val_index = train_test_split(
        range(num_training_samples), test_size=test_size, random_state=seed
    )

    train_dataset = dataset(train_index, split="train")
    val_dataset = dataset(val_index, split="val")

    if dataset == CFMDataset:
        dataloader = TorchDataLoader
    else:
        dataloader = GeometricDataLoader

    train_loader = dataloader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    val_loader = dataloader(
        val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return train_loader, val_loader


def get_test_loader(batch_size=32, dataset=CFMDataset, num_workers=4):
    if dataset == CFMDataset:
        dataloader = TorchDataLoader
    else:
        dataloader = GeometricDataLoader

    num_test_samples = 81600
    test_index = range(num_test_samples)
    test_dataset = dataset(test_index, split="test")
    test_loader = dataloader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return test_loader
