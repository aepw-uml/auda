from typing import Self

import numpy as np
from sklearn.utils.validation import check_array, check_X_y

from .model import Regression


class NaivePersistence(Regression):
    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        X, y = check_X_y(X, y)

        if X.shape[1] != 1:
            raise ValueError('NaivePersistenceModel expects a single feature.')

        order = np.argsort(X[:, 0])
        self.parameters['last_y'] = float(y[order][-1])
        self.last_y = self.parameters['last_y']

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = check_array(X)

        if X.shape[1] != 1:
            raise ValueError('NaivePersistenceModel expects a single feature.')

        m = X.shape[0]

        return np.full((m,), self.parameters['last_y'], dtype=float)
