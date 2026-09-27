from typing import Self, override

import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing as EST
from step.model.annual_series import annual_training_series
from step.model.model import SupervisedLearningModel


class ExponentialSmoothing(SupervisedLearningModel):
    """Forecast future values with additive Holt exponential smoothing."""

    @override
    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Fits Holt smoothing after filling internal training-year gaps.

        Missing years are linearly interpolated between observed training
        values. No test observations enter this preprocessing step.

        Args:
            X: Training time indices with shape ``(n_samples, 1)``.
            y: Training targets with shape ``(n_samples,)``.

        Returns:
            The fitted estimator.
        """

        super().fit(X, y, num_features=1)

        X_sorted, y_sorted = annual_training_series(X, y)
        missing = np.isnan(y_sorted)
        self.parameters['missing_years'] = X_sorted[missing].tolist()
        self.parameters['missing_year_method'] = 'training_linear_interpolation'
        if missing.any():
            y_sorted[missing] = np.interp(
                X_sorted[missing], X_sorted[~missing], y_sorted[~missing]
            )

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
        """Forecasts values at future time steps.

        Args:
            X: Future time indices with shape ``(n_samples, 1)``.

        Returns:
            Predicted targets with shape ``(n_samples,)``.

        Raises:
            ValueError: If requested periods are not whole future time steps
                or are not in the future.
        """

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
