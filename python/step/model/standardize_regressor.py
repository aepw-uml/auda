from typing import Any, Self, Type

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from step.model.model import SupervisedLearningModel


class StandardizedRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        regressor_cls: Type[SupervisedLearningModel],
        hyperparameters: dict[str, Any] | None = None,
        regressor_kwargs: dict[str, Any] | None = None,
        use_x_scaler: bool = True,
        use_y_scaler: bool = True,
    ) -> None:
        """Initializes the StandardizedRegressor.

        Args:
            regressor_cls: The class of the underlying regressor to use.
            regressor_kwargs: Optional keyword arguments to pass to the
                regressor during initialization.
            use_x_scaler: Whether to apply standard scaling to the input
                features.
            use_y_scaler: Whether to apply standard scaling to the target
                variable.
        """

        self.regressor_cls: Type = regressor_cls
        self.hyperparameters: dict[str, Any] = hyperparameters or {}
        self.regressor_kwargs: dict[str, Any] = regressor_kwargs or {}
        self.use_x_scaler: bool = use_x_scaler
        self.use_y_scaler: bool = use_y_scaler

        self.regressor_: Any = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Fits the StandardizedRegressor to the training data.

        Args:
            X: Training features with shape (n_samples, n_features).
            y: Training targets with shape (n_samples,).
        """

        X_fit: np.ndarray = X
        y_fit: np.ndarray = y

        if self.use_x_scaler:
            self.x_scaler_ = StandardScaler()
            X_fit = self.x_scaler_.fit_transform(X)

        if self.use_y_scaler:
            self.y_scaler_ = StandardScaler()
            y_fit = self.y_scaler_.fit_transform(y.reshape(-1, 1)).ravel()

        kwargs = self.regressor_kwargs or {}
        self.regressor_ = self.regressor_cls(self.hyperparameters, **kwargs)
        self.regressor_.fit(X_fit, y_fit)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts target values for the given input features.

        Args:
            X: Features with shape (n_samples, n_features).
        """

        X_pred = X
        if self.use_x_scaler:
            X_pred = self.x_scaler_.transform(X)

        y_pred = np.asarray(self.regressor_.predict(X_pred), dtype=float)

        if self.use_y_scaler:
            y_pred = self.y_scaler_.inverse_transform(
                y_pred.reshape(-1, 1)
            ).ravel()

        return y_pred
