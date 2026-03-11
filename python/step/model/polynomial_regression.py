from typing import Any, Self

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.validation import check_array, check_X_y

from .model import Regression


class PolynomialRegression(Regression):
    def __init__(
        self,
        hyperparameters: dict[str, Any],
        fit_intercept: bool = True,
    ) -> None:
        super().__init__(hyperparameters)
        self.hyperparameters: dict[str, Any] = {
            'degree': int(hyperparameters.get('degree', 2)),
        }

        self.fit_intercept: bool = fit_intercept

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        X, y = check_X_y(X, y)

        if X.shape[1] != 1:
            raise ValueError('Polynomial regression expects a single feature.')

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
        X = check_array(X)

        if X.shape[1] != 1:
            raise ValueError('Polynomial regression expects a single feature.')

        X_poly = self.polynomial_features_.transform(X)
        return self.regressor_.predict(X_poly)
