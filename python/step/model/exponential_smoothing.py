from typing import Self, override

import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing as EST

from .model import SupervisedLearningModel


class ExponentialSmoothing(SupervisedLearningModel):
    @override
    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        super().fit(X, y, num_features=1)

        order = np.argsort(X[:, 0])
        X_sorted = X[order, 0]
        y_sorted = y[order]

        self.parameters['last_x'] = float(X_sorted[-1])
        self.model_ = EST(
            y_sorted,
            trend='add',
            seasonal=None,
            initialization_method='estimated',
        ).fit()

        return self

    @override
    def predict(self, X: np.ndarray) -> np.ndarray:
        super().predict(X, num_features=1)

        X = np.asarray(X, dtype=float)
        t: np.ndarray = X[:, 0]

        steps = t - self.parameters['last_x']
        if not np.allclose(steps, np.round(steps)):
            raise ValueError(
                'ExponentialSmoothing expects forecast periods aligned to '
                'whole time steps.'
            )

        steps = np.round(steps).astype(int)
        if np.any(steps <= 0):
            raise ValueError(
                'ExponentialSmoothing can only forecast future time steps.'
            )

        max_step: int = steps.max()
        forecast = self.model_.forecast(steps=max_step)
        y_hat = forecast[steps - 1]

        return np.asarray(y_hat, dtype=float)
