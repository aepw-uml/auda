from typing import Self

import numpy as np
from step.model.model import SupervisedLearningModel


class NaivePersistence(SupervisedLearningModel):
    """Predict every future value as the last observed target."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Fit the persistence baseline.

        Args:
            X: Training time indices with shape ``(n_samples, 1)``.
            y: Training targets with shape ``(n_samples,)``.

        Returns:
            The fitted estimator.
        """
        super().fit(X, y, num_features=1)

        order = np.argsort(X[:, 0])
        self.parameters['last_y'] = float(y[order][-1])

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the last observed target for each input row.

        Args:
            X: Feature matrix with shape ``(n_samples, 1)``.

        Returns:
            Predicted targets with shape ``(n_samples,)``.
        """
        super().predict(X, num_features=1)

        m = X.shape[0]
        return np.full((m,), self.parameters['last_y'], dtype=float)
