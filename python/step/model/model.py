from abc import ABC
from typing import Any, Self

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, check_array
from sklearn.utils.validation import check_X_y


class SupervisedLearningModel(BaseEstimator, ABC):
    """Base class for supervised learning models in the project.

    Attributes:
        hyperparameters: Configuration values provided when the model is
            created.
        parameters: Learned model state populated during fitting.
    """

    def __init__(self, hyperparameters: dict[str, Any], **kwargs) -> None:
        """Initialize the model with hyperparameters.

        Args:
            hyperparameters: Configuration values used by subclasses.
            **kwargs: Additional keyword arguments accepted for API
                compatibility.
        """
        self.hyperparameters: dict[str, Any] = hyperparameters
        self.parameters: dict[str, Any] = {}
        _ = kwargs

    def fit(
        self, X: np.ndarray, y: np.ndarray, num_features: int | None = None
    ) -> Self:
        """Validate training data before subclass-specific fitting.

        Args:
            X: Training features with shape ``(n_samples, n_features)``.
            y: Training targets with shape ``(n_samples,)``.
            num_features: Expected number of features in ``X``.

        Returns:
            The model instance.

        Raises:
            ValueError: If the input data does not match the expected feature
                count.
        """
        X, y = check_X_y(X, y)
        if num_features is not None and X.shape[1] != num_features:
            raise ValueError(
                f'Expected {num_features} features, got {X.shape[1]}.'
            )

        return self

    def predict(
        self, X: np.ndarray, num_features: int | None = None
    ) -> np.ndarray:
        """Validate inference data before subclass-specific prediction.

        Args:
            X: Features with shape ``(n_samples, n_features)``.
            num_features: Expected number of features in ``X``.

        Returns:
            The validated feature array.

        Raises:
            ValueError: If the input data does not match the expected feature
                count.
        """
        X = check_array(X)

        if num_features is not None and X.shape[1] != num_features:
            raise ValueError(
                f'Expected {num_features} features, got {X.shape[1]}.'
            )

        return X

    def __sklearn_is_fitted__(self) -> bool:
        """Return whether the estimator has learned parameters.

        Returns:
            ``True`` when fitting has populated model parameters.
        """
        return self.parameters != {}


class Regression(SupervisedLearningModel, RegressorMixin):
    """Base class for regression estimators."""

    pass
