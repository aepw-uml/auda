from typing import Self, override

import numpy as np
from step.model.model import SupervisedLearningModel


class DriftBaseline(SupervisedLearningModel):
    @override
    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        super().fit(X, y, num_features=1)

        order = np.argsort(X[:, 0])
        X = X[order, 0]
        y = y[order]

        x0, y0 = float(X[0]), float(y[0])
        x1, y1 = float(X[-1]), float(y[-1])

        denominator = x1 - x0
        slope = (y1 - y0) / denominator if abs(denominator) > 1e-12 else 0.0

        self.parameters['x0'] = x0
        self.parameters['y0'] = y0
        self.parameters['slope'] = slope

        return self

    @override
    def predict(self, X: np.ndarray) -> np.ndarray:
        super().predict(X, num_features=1)

        x0 = self.parameters['x0']
        y0 = self.parameters['y0']
        slope = self.parameters['slope']
        X = np.asarray(X, dtype=float)
        t = X[:, 0]

        return (y0 + slope * (t - x0)).astype(float)
