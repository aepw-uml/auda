from typing import Any, Self, Type

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler


class StandardizedRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        regressor_cls: Type,
        regressor_kwargs: dict[str, Any] | None = None,
        use_x_scaler: bool = True,
        use_y_scaler: bool = True,
    ) -> None:
        self.regressor_cls: Type = regressor_cls
        self.regressor_kwargs: dict[str, Any] = regressor_kwargs or {}
        self.use_x_scaler: bool = use_x_scaler
        self.use_y_scaler: bool = use_y_scaler

        self.regressor_: Any = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        X_fit: np.ndarray = X
        y_fit: np.ndarray = y

        if self.use_x_scaler:
            self.x_scaler_ = StandardScaler()
            X_fit = self.x_scaler_.fit_transform(X)

        if self.use_y_scaler:
            self.y_scaler_ = StandardScaler()
            y_fit = self.y_scaler_.fit_transform(y.reshape(-1, 1)).ravel()

        kwargs = self.regressor_kwargs or {}
        self.regressor_ = self.regressor_cls(**kwargs)
        self.regressor_.fit(X_fit, y_fit)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_pred = X
        if self.use_x_scaler:
            X_pred = self.x_scaler_.transform(X)

        y_pred = np.asarray(self.regressor_.predict(X_pred), dtype=float)

        if self.use_y_scaler:
            y_pred = self.y_scaler_.inverse_transform(
                y_pred.reshape(-1, 1)
            ).ravel()

        return y_pred
