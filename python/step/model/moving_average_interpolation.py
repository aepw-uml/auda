from typing import Any, Self, override

import numpy as np
from step.model.model import Regression


class MovingAverageInterpolation(Regression):
    """Interpolates one-dimensional data with a local moving average."""

    def __init__(
        self,
        hyperparameters: dict[str, Any],
        **kwargs,
    ) -> None:
        """Initializes the moving average interpolation model.

        Args:
            hyperparameters: Model configuration containing ``window_size``.
            **kwargs: Additional keyword arguments forwarded to the base
                class.
        """

        super().__init__(hyperparameters, **kwargs)
        self.hyperparameters: dict[str, Any] = {
            'window_size': int(hyperparameters.get('window_size', 3)),
        }

    @override
    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Fits the moving average interpolation model.

        Args:
            X: Training coordinates with shape ``(n_samples, 1)``.
            y: Training targets with shape ``(n_samples,)``.

        Returns:
            The fitted estimator.

        Raises:
            ValueError: If ``window_size`` is smaller than 1.
        """

        super().fit(X, y, num_features=1)

        window_size = self.hyperparameters['window_size']
        if window_size < 1:
            raise ValueError(
                'MovingAverageInterpolation window_size must be at least 1.'
            )

        order = np.argsort(X[:, 0])
        self.x_train_ = np.asarray(X[order, 0], dtype=float)
        self.y_train_ = np.asarray(y[order], dtype=float)
        self.window_size_ = min(window_size, self.x_train_.size)

        self.parameters['window_size'] = int(self.window_size_)
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
        interpolated = np.empty_like(x_query, dtype=float)
        window_radius = self.window_size_ // 2

        for index, x_value in enumerate(x_query):
            insertion_index = int(np.searchsorted(self.x_train_, x_value))
            start_index = max(0, insertion_index - window_radius)
            end_index = start_index + self.window_size_

            if end_index > self.x_train_.size:
                end_index = self.x_train_.size
                start_index = end_index - self.window_size_

            interpolated[index] = np.mean(self.y_train_[start_index:end_index])

        return interpolated.astype(float)
