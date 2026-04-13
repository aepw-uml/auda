from typing import Any, Self

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from step.model.model import Regression


class PolynomialRegression(Regression):
    """Regression model that expands one feature into polynomial terms."""

    def __init__(
        self,
        hyperparameters: dict[str, Any],
        fit_intercept: bool = True,
        **kwargs,
    ) -> None:
        """Initializes the polynomial regression model.

        Args:
            hyperparameters: Model configuration, including ``degree``.
            fit_intercept: Whether to fit an intercept term.
            **kwargs: Additional keyword arguments forwarded to the base class.
        """

        super().__init__(hyperparameters, **kwargs)
        self.hyperparameters: dict[str, Any] = {
            'degree': int(hyperparameters.get('degree', 2)),
        }
        self.fit_intercept: bool = fit_intercept

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Fits the polynomial regression model.

        Args:
            X: Training features with shape ``(n_samples, 1)``.
            y: Training targets with shape ``(n_samples,)``.

        Returns:
            The fitted estimator.

        Raises:
            ValueError: If the configured polynomial degree is less than 1.
        """

        super().fit(X, y, num_features=1)

        degree = self.hyperparameters['degree']
        if degree < 1:
            raise ValueError('Polynomial regression degree must be at least 1.')

        self.polynomial_features_ = PolynomialFeatures(
            degree=degree,
            include_bias=False,
        )
        X_poly = self.polynomial_features_.fit_transform(X)

        self.regressor_ = LinearRegression(fit_intercept=self.fit_intercept)
        self.regressor_.fit(X_poly, y)

        self.parameters['coefficients'] = self.regressor_.coef_
        if self.fit_intercept:
            self.parameters['intercept'] = self.regressor_.intercept_

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts targets for new samples.

        Args:
            X: Feature matrix with shape ``(n_samples, 1)``.

        Returns:
            Predicted targets with shape ``(n_samples,)``.
        """

        super().predict(X, num_features=1)

        X_poly = self.polynomial_features_.transform(X)
        return self.regressor_.predict(X_poly)
