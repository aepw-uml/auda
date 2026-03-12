from typing import Self

import numpy as np
from step.model.model import SupervisedLearningModel


class NaivePersistence(SupervisedLearningModel):
    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        super().fit(X, y, num_features=1)

        order = np.argsort(X[:, 0])
        self.parameters['last_y'] = float(y[order][-1])

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        super().predict(X, num_features=1)

        m = X.shape[0]
        return np.full((m,), self.parameters['last_y'], dtype=float)
