import numpy as np
import pandas as pd
from data_cache import numpy_cache
from scipy import stats
from sklearn.preprocessing import LabelEncoder


@numpy_cache
def load_data(
    dummy=False, shuffle=True, seed=42, normalize=True, filter=True, resize=False
):
    """features : {0: "venue", 1: "order_id", 2: "action", 3: "side", 4: "price", 5: "bid", 6: "ask", 7: "bid_size", 8: "ask_size", 9: "trade", 10: "flux"}"""
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

    # REMOVE OUTLIERS
    if filter:
        features_to_filter = ["price", "bid", "ask", "bid_size", "ask_size", "flux"]
        to_remove = []
        for f in features_to_filter:
            to_remove.append(X[(np.abs(stats.zscore(X[f])) >= 8)]["obs_id"])
        to_remove = pd.concat(to_remove)
        X = X[~X["obs_id"].isin(to_remove)]
        y = y[~y["obs_id"].isin(to_remove)]

    # NORMALIZE
    if normalize:
        # normalize by group the features that have the same dimension
        normalize_log = ["bid_size", "ask_size", "flux"]
        for feature in normalize_log:
            X[feature] = np.sign(X[feature]) * np.log(np.abs(X[feature]) + 1)
            X_test[feature] = np.sign(X_test[feature]) * np.log(
                np.abs(X_test[feature]) + 1
            )

        normalize_groups_minmax = [
            ["bid_size", "ask_size"],
        ]
        for group in normalize_groups_minmax:
            mini = 1e32
            maxi = -1e32
            for feature in group:
                mini = min(mini, X[feature].min())
                maxi = max(maxi, X[feature].max())
            for feature in group:
                X[feature] = (X[feature] - mini) / (maxi - mini)
                X_test[feature] = (X_test[feature] - mini) / (maxi - mini)

        normalize_groups_standard = []
        for group in normalize_groups_standard:
            mean = 0
            std = 0
            for feature in group:
                mean += X[feature].mean()
                std += X[feature].std()
            mean /= len(group)
            std /= len(group)
            for feature in group:
                X[feature] = (X[feature] - mean) / std
                X_test[feature] = (X_test[feature] - mean) / std

    # NUMPY AND RESHAPE
    X = X.to_numpy()
    X = X.reshape(-1, 100, X.shape[1])
    X = X[:, :, 1:]
    X_test = X_test.to_numpy()
    X_test = X_test.reshape(-1, 100, X_test.shape[1])
    X_test = X_test[:, :, 1:]

    # Y
    y = y.set_index("obs_id").to_numpy().reshape(-1)

    # RESIZE
    if resize:
        # Adapt test data so that its distribution is similar to the train data

        # ask
        train_ask = X[:, :, 6].reshape(-1)
        test_ask = X_test[:, :, 6].reshape(-1)
        test_spread = np.max(train_ask) - np.min(train_ask)
        train_spread = np.max(test_ask) - np.min(test_ask)
        test_ask = (
            test_ask - np.min(test_ask)
        ) / train_spread * test_spread / 1.8 + np.min(test_ask)
        X_test[:, :, 6] = test_ask.reshape(-1, 100)

    # SHUFFLE
    if shuffle:
        shuffled_index = np.random.permutation(X.shape[0])
    else:
        shuffled_index = np.arange(X.shape[0])

    return X[shuffled_index, :, :], y[shuffled_index], X_test
