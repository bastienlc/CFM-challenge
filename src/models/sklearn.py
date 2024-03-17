import os
import pickle

from sklearn.base import ClassifierMixin
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import (
    LogisticRegression,
    PassiveAggressiveClassifier,
    Perceptron,
    RidgeClassifier,
)

verbose = 1


class Voting(ClassifierMixin):
    def __init__(self) -> None:
        super().__init__()

        self.base_classifier = VotingClassifier(
            estimators=[
                ("ada", AdaBoostClassifier(algorithm="SAMME", n_estimators=100)),
                ("rf", RandomForestClassifier()),
                ("et", ExtraTreesClassifier()),
                ("lr", LogisticRegression()),
                ("pa", PassiveAggressiveClassifier()),
                ("p", Perceptron()),
                ("rc", RidgeClassifier()),
            ],
            verbose=verbose,
            n_jobs=20,
        )

    def fit(self, X, y):
        self.base_classifier.fit(X, y)

    def predict(self, X):
        return self.base_classifier.predict(X)

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, "base_classifier.pkl"), "wb") as f:
            pickle.dump(self.base_classifier, f)

    def load(self, path):
        with open(os.path.join(path, "base_classifier.pkl"), "rb") as f:
            self.base_classifier = pickle.load(f)
