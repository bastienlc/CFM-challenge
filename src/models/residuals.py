import os
import pickle

import numpy as np
from sklearnex import patch_sklearn

patch_sklearn()

from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor

verbose = 0


class ResidualModel(ClassifierMixin):
    def __init__(self, n_jobs=20) -> None:
        super().__init__()
        self.n_jobs = n_jobs

        self.base_classifier = RandomForestClassifier(
            verbose=verbose,
            n_jobs=self.n_jobs,
            max_depth=14,
        )
        self.regressors = []
        for _ in range(24):
            self.regressors.append(
                [
                    ("Ridge", Ridge(n_jobs=self.n_jobs)),
                    ("KNeighborsRegressor", KNeighborsRegressor(n_jobs=self.n_jobs)),
                    ("LinearRegression", LinearRegression(n_jobs=self.n_jobs)),
                ]
            )
        self.stacking_regressors = []
        for k in range(24):
            self.stacking_regressors.append(
                StackingRegressor(
                    estimators=self.regressors[k],
                    final_estimator=LinearRegression(n_jobs=self.n_jobs),
                )
            )

    def fit(self, X, y):
        print("Training base classifier...")
        self.base_classifier.fit(X, y)
        base_prediction = self.base_classifier.predict_proba(X)
        print("Base classifier accuracy", (base_prediction.argmax(1) == y).mean())

        y_train_residuals = np.eye(np.max(y) + 1)[y] - base_prediction

        for k in range(24):
            print(f"Training regressor {k}...")
            self.stacking_regressors[k].fit(X, y_train_residuals[:, k])

    def predict_proba(self, X):
        base_pred = self.base_classifier.predict_proba(X)
        residuals = np.zeros_like(base_pred)
        for k in range(24):
            residuals[:, k] = self.stacking_regressors[k].predict(X)
        return base_pred + residuals

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, "base_classifier.pkl"), "wb") as f:
            pickle.dump(self.base_classifier, f)

        for k in range(24):
            with open(os.path.join(path, f"stacking_regressor_{k}.pkl"), "wb") as f:
                pickle.dump(self.stacking_regressors[k], f)

    def load(self, path):
        with open(os.path.join(path, "base_classifier.pkl"), "rb") as f:
            self.base_classifier = pickle.load(f)

        for k in range(24):
            with open(os.path.join(path, f"stacking_regressor_{k}.pkl"), "rb") as f:
                self.stacking_regressors[k] = pickle.load(f)
