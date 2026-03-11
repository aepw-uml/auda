from abc import ABC, abstractmethod
from typing import Any, Self

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class Regression(RegressorMixin, BaseEstimator, ABC):
    def __init__(self, hyperparameters: dict[str, Any]) -> None:
        self.hyperparameters: dict[str, Any] = hyperparameters
        self.parameters: dict[str, Any] = {}

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def __sklearn_is_fitted__(self) -> bool:
        return self.parameters != {}
