from typing import Any, Self, override

import numpy as np
from step.model.model import Regression


class LinearInterpolation(Regression):
    """Interpolates one-dimensional data with piecewise linear segments."""

    def __init__(
        self,
        hyperparameters: dict[str, Any],
        **kwargs,
    ) -> None:
        """Initializes the linear interpolation model.

        Args:
            hyperparameters: Model configuration accepted for API
                compatibility.
            **kwargs: Additional keyword arguments forwarded to the base
                class.
        """

        super().__init__(hyperparameters, **kwargs)

    @override
    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Fits the linear interpolation model.

        Args:
            X: Training coordinates with shape ``(n_samples, 1)``.
            y: Training targets with shape ``(n_samples,)``.

        Returns:
            The fitted estimator.

        Raises:
            ValueError: If fewer than two unique x-values are provided.
        """

        super().fit(X, y, num_features=1)

        order = np.argsort(X[:, 0])
        x_sorted = np.asarray(X[order, 0], dtype=float)
        y_sorted = np.asarray(y[order], dtype=float)
        unique_x, unique_indices = np.unique(x_sorted, return_index=True)

        if unique_x.size < 2:
            raise ValueError(
                'LinearInterpolation requires at least two unique x-values.'
            )

        self.x_train_ = unique_x
        self.y_train_ = y_sorted[unique_indices]
        self.parameters['x_min'] = float(self.x_train_[0])
        self.parameters['x_max'] = float(self.x_train_[-1])
        self.parameters['num_points'] = int(self.x_train_.size)

        return self

    @override
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Interpolates targets for new coordinates.

        Args:
            X: Query coordinates with shape ``(n_samples, 1)``.

        Returns:
            Interpolated targets with shape ``(n_samples,)``.
        """

        super().predict(X, num_features=1)

        x_query = np.asarray(X[:, 0], dtype=float)
        return np.interp(x_query, self.x_train_, self.y_train_).astype(float)
