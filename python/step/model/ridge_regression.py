from typing import Any, Self

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from step.model.model import Regression


class RidgeRegression(Regression):
    """Polynomial regression model with L2 regularization."""

    def __init__(
        self,
        hyperparameters: dict[str, Any],
        fit_intercept: bool = True,
        **kwargs,
    ) -> None:
        """Initializes the polynomial ridge regression model.

        Args:
            hyperparameters: Model configuration, including ``degree`` and
                ``alpha``.
            fit_intercept: Whether to fit an intercept term.
            **kwargs: Additional keyword arguments forwarded to the base class.
        """

        super().__init__(hyperparameters, **kwargs)

        self.hyperparameters: dict[str, Any] = {
            'degree': int(hyperparameters.get('degree', 2)),
            'alpha': float(hyperparameters.get('alpha', 1.0)),
        }
        self.fit_intercept: bool = fit_intercept

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Fits the polynomial ridge regression model.

        Args:
            X: Training features with shape ``(n_samples, 1)``.
            y: Training targets with shape ``(n_samples,)``.

        Returns:
            The fitted estimator.

        Raises:
            ValueError: If ``degree`` is less than 1.
            ValueError: If ``alpha`` is negative.
        """

        super().fit(X, y, num_features=1)

        degree = self.hyperparameters['degree']
        alpha = self.hyperparameters['alpha']
        if degree < 1:
            raise ValueError(
                'Polynomial ridge regression degree must be at least 1.'
            )
        if alpha < 0.0:
            raise ValueError('Ridge regression alpha must be non-negative.')

        self.regressor_ = Pipeline(
            [
                (
                    'PolynomialFeatures',
                    PolynomialFeatures(
                        degree=degree,
                        include_bias=False,
                    ),
                ),
                (
                    'Ridge',
                    Ridge(
                        alpha=alpha,
                        fit_intercept=self.fit_intercept,
                    ),
                ),
            ]
        )
        self.regressor_.fit(X, y)

        self.polynomial_features_ = self.regressor_.named_steps[
            'PolynomialFeatures'
        ]
        ridge_model = self.regressor_.named_steps['Ridge']

        self.parameters['coefficients'] = ridge_model.coef_
        if self.fit_intercept:
            self.parameters['intercept'] = ridge_model.intercept_

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts targets for new samples.

        Args:
            X: Feature matrix with shape ``(n_samples, 1)``.

        Returns:
            Predicted targets with shape ``(n_samples,)``.
        """

        super().predict(X, num_features=1)

        return self.regressor_.predict(X)
