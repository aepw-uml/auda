from typing import Any, Self, override

import numpy as np
from step.model.model import Regression


class CubicSpline(Regression):
    """Interpolates one-dimensional data with a natural cubic spline."""

    def __init__(
        self,
        hyperparameters: dict[str, Any],
        **kwargs,
    ) -> None:
        """Initializes the cubic spline interpolation model.

        Args:
            hyperparameters: Model configuration accepted for API
                compatibility.
            **kwargs: Additional keyword arguments forwarded to the base
                class.
        """

        super().__init__(hyperparameters, **kwargs)

    @override
    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Fits the natural cubic spline coefficients.

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
                'CubicSpline requires at least two unique x-values.'
            )

        self.x_train_ = unique_x
        self.y_train_ = y_sorted[unique_indices]
        interval_widths = np.diff(self.x_train_)
        num_points = self.x_train_.size

        second_derivatives = np.zeros(num_points, dtype=float)
        if num_points > 2:
            system = np.zeros((num_points - 2, num_points - 2), dtype=float)
            rhs = np.zeros(num_points - 2, dtype=float)

            for row_index in range(num_points - 2):
                left_width = interval_widths[row_index]
                right_width = interval_widths[row_index + 1]

                system[row_index, row_index] = 2.0 * (left_width + right_width)
                if row_index > 0:
                    system[row_index, row_index - 1] = left_width
                if row_index < num_points - 3:
                    system[row_index, row_index + 1] = right_width

                rhs[row_index] = 6.0 * (
                    (
                        self.y_train_[row_index + 2]
                        - self.y_train_[row_index + 1]
                    )
                    / right_width
                    - (self.y_train_[row_index + 1] - self.y_train_[row_index])
                    / left_width
                )

            second_derivatives[1:-1] = np.linalg.solve(system, rhs)

        self.second_derivatives_ = second_derivatives
        self.parameters['x_min'] = float(self.x_train_[0])
        self.parameters['x_max'] = float(self.x_train_[-1])
        self.parameters['num_points'] = int(num_points)

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

        for index, x_value in enumerate(x_query):
            if x_value <= self.x_train_[0]:
                interval_index = 0
            elif x_value >= self.x_train_[-1]:
                interval_index = self.x_train_.size - 2
            else:
                interval_index = (
                    np.searchsorted(self.x_train_, x_value, side='right') - 1
                )

            left_x = self.x_train_[interval_index]
            right_x = self.x_train_[interval_index + 1]
            width = right_x - left_x
            left_weight = (right_x - x_value) / width
            right_weight = (x_value - left_x) / width

            interpolated[index] = (
                left_weight * self.y_train_[interval_index]
                + right_weight * self.y_train_[interval_index + 1]
                + (
                    (left_weight**3 - left_weight)
                    * self.second_derivatives_[interval_index]
                    + (right_weight**3 - right_weight)
                    * self.second_derivatives_[interval_index + 1]
                )
                * (width**2)
                / 6.0
            )

        return interpolated.astype(float)
