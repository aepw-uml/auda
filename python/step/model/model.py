from abc import ABC
from typing import Any, Self

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, check_array
from sklearn.utils.validation import check_X_y


class SupervisedLearningModel(BaseEstimator, ABC):
    def __init__(self, hyperparameters: dict[str, Any], **kwargs) -> None:
        self.hyperparameters: dict[str, Any] = hyperparameters
        self.parameters: dict[str, Any] = {}
        _ = kwargs

    def fit(
        self, X: np.ndarray, y: np.ndarray, num_features: int | None = None
    ) -> Self:
        X, y = check_X_y(X, y)
        if num_features is not None and X.shape[1] != num_features:
            raise ValueError(
                f'Expected {num_features} features, got {X.shape[1]}.'
            )

        return self

    def predict(
        self, X: np.ndarray, num_features: int | None = None
    ) -> np.ndarray:
        X = check_array(X)

        if num_features is not None and X.shape[1] != num_features:
            raise ValueError(
                f'Expected {num_features} features, got {X.shape[1]}.'
            )

        return X

    def __sklearn_is_fitted__(self) -> bool:
        return self.parameters != {}


class Regression(SupervisedLearningModel, RegressorMixin):
    pass
