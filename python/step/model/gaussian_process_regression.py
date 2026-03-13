from typing import Any, Self, override

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from step.model.model import Regression


class GaussianProcessRegression(Regression):
    """Gaussian process regressor with an RBF kernel and additive noise.

    The model learns a smooth latent function with an RBF kernel while a
    white-noise term captures observation noise. Hyperparameters are read from
    ``hyperparameters`` and stored as floats so they are safe to pass directly
    to scikit-learn.

    Attributes:
        hyperparameters: Model configuration containing ``length_scale`` and
            ``noise_level``.
    """

    def __init__(
        self,
        hyperparameters: dict[str, Any],
        **kwargs,
    ) -> None:
        """Initializes the Gaussian process regression model.

        Args:
            hyperparameters: Model configuration containing ``length_scale``
                and ``noise_level``.
            **kwargs: Additional keyword arguments forwarded to the base class.
        """

        super().__init__(hyperparameters, **kwargs)
        self.hyperparameters: dict[str, Any] = {
            'length_scale': float(hyperparameters.get('length_scale', 1.0)),
            'noise_level': float(hyperparameters.get('noise_level', 1e-2)),
        }

    @override
    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Fits the Gaussian process regressor.

        Args:
            X: Training features with shape ``(n_samples, n_features)``.
            y: Training targets with shape ``(n_samples,)``.

        Returns:
            The fitted estimator.

        Raises:
            ValueError: If ``length_scale`` is not positive or
                ``noise_level`` is negative.
        """

        super().fit(X, y)

        length_scale = self.hyperparameters['length_scale']
        noise_level = self.hyperparameters['noise_level']

        if length_scale <= 0.0:
            raise ValueError(
                'Gaussian process regression length_scale must be positive.'
            )
        if noise_level < 0.0:
            raise ValueError(
                'Gaussian process regression noise_level must be non-negative.'
            )

        kernel = RBF(
            length_scale=length_scale,
            length_scale_bounds=(1e-10, 1e3),
        ) + WhiteKernel(
            noise_level=noise_level,
            noise_level_bounds=(1e-6, 1e1),
        )

        self.regressor_ = GaussianProcessRegressor(
            kernel=kernel,
            optimizer=None,  # type: ignore[assignment]
        )
        self.regressor_.fit(X, y)

        self.parameters['kernel'] = self.regressor_.kernel_
        self.parameters['length_scale'] = length_scale
        self.parameters['noise_level'] = noise_level

        return self

    @override
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts targets for new samples.

        Args:
            X: Feature matrix with shape ``(n_samples, n_features)``.

        Returns:
            Predicted targets with shape ``(n_samples,)``.
        """

        super().predict(X)

        return np.asarray(self.regressor_.predict(X), dtype=float)
