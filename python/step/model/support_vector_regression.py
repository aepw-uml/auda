from typing import Any, Self

import numpy as np
from step.model.model import Regression


class SupportVectorRegression(Regression):
    def __init__(
        self,
        hyperparameters: dict[str, Any],
        kernel: str = 'rbf',
        **kwargs,
    ) -> None:
        """Initializes the ridge regression model.

        Args:
            hyperparameters: Model configuration, including ``alpha``.
            kernel: The kernel type to be used in the algorithm.
            **kwargs: Additional keyword arguments forwarded to the base class.
        """

        super().__init__(hyperparameters, **kwargs)

        self.hyperparameters: dict[str, Any] = {
            'C': float(hyperparameters.get('C', 0.1)),
            'epsilon': float(hyperparameters.get('epsilon', 0.1)),
            'gamma': hyperparameters.get('gamma', 'scale'),
        }
        self.kernel: str = kernel

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Fits the model to the training data.

        Args:
            X: Training data features.
            y: Training data target values.
        """
        from sklearn.svm import SVR

        c = self.hyperparameters['C']
        epsilon = self.hyperparameters['epsilon']
        gamma = self.hyperparameters['gamma']

        self.regressor_ = SVR(
            kernel=self.kernel,
            C=c,
            epsilon=epsilon,
            gamma=gamma,
        )
        self.regressor_.fit(X, y)

        svr = self.regressor_
        num_support_vectors = len(svr.support_.tolist())
        dual_coefficients = (
            svr.dual_coef_.ravel().astype(float).tolist()  # type: ignore
        )
        intercept = float(svr.intercept_[0])
        gamma = svr.gamma

        self.parameters['num_support_vectors'] = num_support_vectors
        self.parameters['dual_coefficients'] = dual_coefficients
        self.parameters['intercept'] = intercept
        self.parameters['gamma'] = gamma

        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predicts target values for the given input data.

        Args:
            X: Input data features.

        Returns:
            Predicted target values.
        """

        return self.regressor_.predict(X)
