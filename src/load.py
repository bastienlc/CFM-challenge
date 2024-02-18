import numpy as np
import pandas as pd
import torch
from data_cache import numpy_cache
from sklearn.preprocessing import LabelEncoder


@numpy_cache
def load_data(dummy=False, shuffle=True, seed=42):
    np.random.seed(seed)
    # LOAD
    if dummy:
        X = pd.read_csv("data/small_x_train.csv")
        y = pd.read_csv("data/small_y_train.csv")
        X_test = pd.read_csv("data/small_x_train.csv")
    else:
        X = pd.read_csv("data/x_train.csv")
        y = pd.read_csv("data/y_train.csv")
        X_test = pd.read_csv("data/x_test.csv")

    # LABELS
    labels_to_encode = ["action", "side", "trade"]
    for label in labels_to_encode:
        label_encoder = LabelEncoder()
        X[label] = label_encoder.fit_transform(X[label])
        X_test[label] = label_encoder.transform(X_test[label])

    # FILL
    fill_values = X.mean()
    nb = X.isna().sum().sum()
    if nb > 0:
        print("Filling", nb, "NAs in train")
    X = X.fillna(fill_values)

    nb = X_test.isna().sum().sum()
    if nb > 0:
        print("Filling", nb, "NAs in test")
    X_test = X_test.fillna(fill_values)

    # NUMPY AND RESHAPE
    X = X.to_numpy()
    X = X.reshape(-1, 100, X.shape[1])
    X = X[:, :, 1:]
    X_test = X_test.to_numpy()
    X_test = X_test.reshape(-1, 100, X_test.shape[1])
    X_test = X_test[1:, :, :]

    # Y
    y = y.set_index("obs_id").to_numpy().reshape(-1)

    # SHUFFLE
    if shuffle:
        shuffled_index = np.random.permutation(X.shape[0])
    else:
        shuffled_index = np.arange(X.shape[0])

    return X[shuffled_index, :, :], y[shuffled_index], X_test


def get_data_loaders(X_train, y_train, X_val, y_val, device, batch_size=32):
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.LongTensor(y_train).to(device)

    X_val = torch.FloatTensor(X_val).to(device)
    y_val = torch.LongTensor(y_val).to(device)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )

    return train_loader, val_loader
