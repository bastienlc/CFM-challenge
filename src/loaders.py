from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Dataset as GeometricDataset
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
    train_dataset = dataset(split="train", test_size=test_size, seed=seed)
    val_dataset = dataset(split="val", test_size=test_size, seed=seed)

    if issubclass(dataset, GeometricDataset):
        dataloader = GeometricDataLoader
    elif issubclass(dataset, TorchDataset):
        dataloader = TorchDataLoader
    else:
        raise ValueError("Dataset type not recognized")

    train_loader = dataloader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    val_loader = dataloader(
        val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return train_loader, val_loader


def get_test_loader(batch_size=32, dataset=CFMDataset, num_workers=4, shuffle=False):
    if issubclass(dataset, GeometricDataset):
        dataloader = GeometricDataLoader
    elif issubclass(dataset, TorchDataset):
        dataloader = TorchDataLoader
    else:
        raise ValueError("Dataset type not recognized")

    test_dataset = dataset(split="test")
    test_loader = dataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return test_loader
