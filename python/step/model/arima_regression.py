from typing import Any, Self, override

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from step.model.model import SupervisedLearningModel


class ARIMARegression(SupervisedLearningModel):
    """Forecasts one-dimensional time series with an ARIMA model."""

    def __init__(
        self,
        hyperparameters: dict[str, Any],
        **kwargs,
    ) -> None:
        """Initializes the ARIMA regression model.

        Args:
            hyperparameters: Model configuration containing ``p``, ``d``,
                ``q``, and ``trend``.
            **kwargs: Additional keyword arguments forwarded to the base
                class.
        """

        super().__init__(hyperparameters, **kwargs)
        self.hyperparameters: dict[str, Any] = {
            'p': int(hyperparameters.get('p', 2)),
            'd': int(hyperparameters.get('d', 0)),
            'q': int(hyperparameters.get('q', 1)),
            'trend': str(hyperparameters.get('trend', 'n')),
        }

    @override
    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Fits the ARIMA model on the sorted time series.

        Args:
            X: Training time indices with shape ``(n_samples, 1)``.
            y: Training targets with shape ``(n_samples,)``.

        Returns:
            The fitted estimator.

        Raises:
            ValueError: If the ARIMA orders are invalid.
        """

        super().fit(X, y, num_features=1)

        p = self.hyperparameters['p']
        d = self.hyperparameters['d']
        q = self.hyperparameters['q']
        trend = self.hyperparameters['trend']
        if min(p, d, q) < 0:
            raise ValueError('ARIMA orders p, d, and q must be non-negative.')

        order = np.argsort(X[:, 0])
        X_sorted = X[order, 0]
        y_sorted = y[order]

        self.parameters['last_x'] = float(X_sorted[-1])
        self.parameters['order'] = (p, d, q)
        self.parameters['trend'] = trend
        self.model_ = ARIMA(
            y_sorted,
            order=(p, d, q),
            trend=trend,
        ).fit()
        self.parameters['aic'] = float(self.model_.aic)

        return self

    @override
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Forecasts future values at the requested time steps.

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
                'ARIMARegression expects forecast periods aligned to whole '
                'time steps.'
            )

        steps = np.round(steps).astype(int)
        if np.any(steps <= 0):
            raise ValueError(
                'ARIMARegression can only forecast future time steps.'
            )

        max_step: int = int(steps.max())
        forecast = self.model_.forecast(steps=max_step)
        y_hat = forecast[steps - 1]

        return np.asarray(y_hat, dtype=float)
