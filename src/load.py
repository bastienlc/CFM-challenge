import numpy as np
import pandas as pd
from data_cache import numpy_cache
from scipy import stats
from sklearn.preprocessing import LabelEncoder


# @numpy_cache
def load_data(
    dummy=False,
    seed=42,
    normalize=False,
    filter=True,
    nb_ticks_max=3,
):
    tick_size=0.01
    """features : {0: "venue", 1: "order_id", 2: "action", 3: "side", 4: "price", 5: "bid", 6: "ask", 7: "bid_size", 9: "ask_size", 9: "trade", 10: "flux"}"""
    np.random.seed(seed)

    # LOAD
    print("Loading data...")
    if dummy:
        X = pd.read_parquet("data/small_x_train.csv")
        y = pd.read_parquet("data/small_y_train.csv")
        X_test = pd.read_parquet("data/small_x_train.csv")
    else:
        X = pd.read_parquet("data/X_train.parquet")
        y = pd.read_parquet("data/y_train.parquet")
        X_test = pd.read_parquet("data/X_test.parquet")

    print("Adding basic features...")

    X["bid_ask_spread"] = X["ask"] - X["bid"]
    X["Limit Order"] = (X["price"] == X["bid"]) | (X["price"] == X["ask"])

    X_test["bid_ask_spread"] = X_test["ask"] - X_test["bid"]
    X_test["Limit Order"] = (X_test["price"] == X_test["bid"]) | (X_test["price"] == X_test["ask"])

    print("Encoding data...")
    # LABELS
    labels_to_encode = ["action", "side", "trade"]
    for label in labels_to_encode:
        label_encoder = LabelEncoder()
        X[label] = label_encoder.fit_transform(X[label])
        X_test[label] = label_encoder.transform(X_test[label])

    print("Removing extreme values... (> " + str(nb_ticks_max) + " ticks)")
    if nb_ticks_max is not None:
            dummy_row = X.iloc[0]
            X_len_before = len(X)
            X = X[(X["price"] >= X["bid"] - nb_ticks_max*tick_size) & (X["price"] <= X["ask"] + nb_ticks_max*tick_size)]
            # add the dummy row wherever obs_id is missing
            max_obs_id = 160800
            # array of elemnts between 0 and max_obs_id that are not in X.obs_id.unique()
            missing_obs_id = np.setdiff1d(np.arange(0, max_obs_id), X.obs_id.unique())

            # add missing obs_id to X
            missing_rows = pd.DataFrame([dummy_row]*len(missing_obs_id))
            missing_rows["obs_id"] = missing_obs_id

            # merge X and missing_rows on obs_id
            X = pd.concat([X, missing_rows], axis=0)

            # sort X by obs_id
            X = X.sort_values("obs_id")
            X_len_after = len(X)
            print(f"    [Train set] Filtering out of range prices: {X_len_before - X_len_after} rows removed over {X_len_before} rows", f"({'%.2f' % (100*(X_len_before - X_len_after)/X_len_before)}%)")

            X_test_len_before = len(X_test)
            X_test = X_test[(X_test["price"] >= X_test["bid"] - nb_ticks_max*tick_size) & (X_test["price"] <= X_test["ask"] + nb_ticks_max*tick_size)]
             # add the dummy row wherever obs_id is missing
            max_obs_id = 81600
            # array of elemnts between 0 and max_obs_id that are not in X.obs_id.unique()
            missing_obs_id = np.setdiff1d(np.arange(0, max_obs_id), X_test.obs_id.unique())

            # add missing obs_id to X
            missing_rows = pd.DataFrame([dummy_row]*len(missing_obs_id))
            missing_rows["obs_id"] = missing_obs_id

            # merge X and missing_rows on obs_id
            X_test = pd.concat([X_test, missing_rows], axis=0)

            # sort X by obs_id
            X_test = X_test.sort_values("obs_id")
            X_test_len_after = len(X_test)
            print(f"    [Test set] Filtering out of range prices: {X_test_len_before - X_test_len_after} rows removed over {X_test_len_before} rows", f"({'%.2f' % (100*(X_test_len_before - X_test_len_after)/X_test_len_before)}%)")


    print("Normalizing data..." + ("(SKIPPED)" if not normalize else ""))
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

    print("Converting to numpy arrays...")
    # NUMPY AND RESHAPE
    features = X.columns[1:].values

    X = X.values
    X_obs = X[:,0]
    X = X[:,1:]

    X_test = X_test.values
    X_test_obs = X_test[:,0]
    X_test = X_test[:,1:]

    # Y
    y = y.values[:,1]

    return X, X_obs, y, X_test, X_test_obs, features