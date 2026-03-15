from typing import Any, Self, override

import numpy as np
from sklearn.linear_model import TheilSenRegressor as TSE
from step.model.model import Regression


class TheilSenRegression(Regression):
    """Fits a robust linear trend with Theil-Sen regression."""

    def __init__(
        self,
        hyperparameters: dict[str, Any],
        **kwargs,
    ) -> None:
        """Initializes the Theil-Sen regression model.

        Args:
            hyperparameters: Model configuration containing ``window_size``.
            **kwargs: Additional keyword arguments forwarded to the base
                class.
        """

        super().__init__(hyperparameters, **kwargs)
        self.hyperparameters: dict[str, Any] = {
            'window_size': int(hyperparameters.get('window_size', 7)),
        }

    @override
    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Fits the Theil-Sen model on the most recent training window.

        Args:
            X: Training features with shape ``(n_samples, 1)``.
            y: Training targets with shape ``(n_samples,)``.

        Returns:
            The fitted estimator.

        Raises:
            ValueError: If ``window_size`` is smaller than 2.
        """

        super().fit(X, y, num_features=1)

        window_size = self.hyperparameters['window_size']
        if window_size < 2:
            raise ValueError(
                'Theil-Sen regression window_size must be at least 2.'
            )

        order = np.argsort(X[:, 0])
        X_sorted = np.asarray(X[order], dtype=float)
        y_sorted = np.asarray(y[order], dtype=float)
        effective_window_size = min(window_size, X_sorted.shape[0])

        X_window = X_sorted[-effective_window_size:]
        y_window = y_sorted[-effective_window_size:]

        self.regressor_ = TSE(random_state=0)
        self.regressor_.fit(X_window, y_window)

        self.parameters['window_size'] = int(effective_window_size)
        self.parameters['coefficients'] = self.regressor_.coef_
        self.parameters['intercept'] = float(self.regressor_.intercept_)

        return self

    @override
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts future targets with the fitted robust linear trend.

        Args:
            X: Feature matrix with shape ``(n_samples, 1)``.

        Returns:
            Predicted targets with shape ``(n_samples,)``.
        """

        super().predict(X, num_features=1)

        return np.asarray(self.regressor_.predict(X), dtype=float)
