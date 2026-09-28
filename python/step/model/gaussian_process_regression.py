from typing import Any, Self, override

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, WhiteKernel
from step.model.model import Regression


class GaussianProcessRegression(Regression):
    """Gaussian process regressor with linear, RBF, and noise components.

    A DotProduct component supports linear trends, RBF models smooth local
    variation, and WhiteKernel models observation noise. Scikit-learn
    optimizes the kernel by maximizing log-marginal likelihood on the supplied
    training data. AUDA does not externally search or apply the 1-SE rule.

    Attributes:
        hyperparameters: Model configuration containing ``length_scale`` and
            ``noise_level`` initialization values and optimizer restart count.
    """

    def __init__(
        self,
        hyperparameters: dict[str, Any],
        **kwargs,
    ) -> None:
        """Initializes the Gaussian process regression model.

        Args:
            hyperparameters: Model configuration containing ``length_scale``
                and ``noise_level`` initialization values, plus optional
                ``n_restarts_optimizer`` (default 20) and ``random_state``
                (default 42). The restarts use the fixed seed.
            **kwargs: Additional keyword arguments forwarded to the base class.
        """

        super().__init__(hyperparameters, **kwargs)
        self.hyperparameters: dict[str, Any] = {
            'length_scale': float(hyperparameters.get('length_scale', 1.0)),
            'noise_level': float(hyperparameters.get('noise_level', 1e-2)),
            'n_restarts_optimizer': int(
                hyperparameters.get('n_restarts_optimizer', 20)
            ),
            'random_state': int(hyperparameters.get('random_state', 42)),
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

        kernel = DotProduct(
            sigma_0=1.0,
            sigma_0_bounds=(1e-5, 1e5),
        ) + RBF(
            length_scale=length_scale,
            length_scale_bounds=(1e-10, 1e3),
        ) + WhiteKernel(
            noise_level=noise_level,
            noise_level_bounds=(1e-6, 1e1),
        )

        self.regressor_ = GaussianProcessRegressor(
            kernel=kernel,
            optimizer='fmin_l_bfgs_b',
            n_restarts_optimizer=self.hyperparameters['n_restarts_optimizer'],
            random_state=self.hyperparameters['random_state'],
            alpha=1e-10,
            normalize_y=False,
        )
        self.regressor_.fit(X, y)

        self.parameters['kernel'] = self.regressor_.kernel_
        self.parameters['length_scale'] = float(
            self.regressor_.kernel_.k1.k2.length_scale
        )
        self.parameters['noise_level'] = float(
            self.regressor_.kernel_.k2.noise_level
        )
        self.parameters['initial_sigma_0'] = 1.0
        self.parameters['sigma_0'] = float(
            self.regressor_.kernel_.k1.k1.sigma_0
        )
        self.parameters['sigma_0_bounds'] = kernel.k1.k1.sigma_0_bounds
        self.parameters['initial_length_scale'] = length_scale
        self.parameters['initial_noise_level'] = noise_level
        self.parameters['length_scale_bounds'] = (
            kernel.k1.k2.length_scale_bounds
        )
        self.parameters['noise_level_bounds'] = kernel.k2.noise_level_bounds
        self.parameters['optimizer'] = self.regressor_.optimizer
        self.parameters['n_restarts_optimizer'] = (
            self.regressor_.n_restarts_optimizer
        )
        self.parameters['random_state'] = self.regressor_.random_state
        self.parameters['alpha'] = self.regressor_.alpha
        self.parameters['normalize_y'] = self.regressor_.normalize_y
        self.parameters['log_marginal_likelihood'] = float(
            self.regressor_.log_marginal_likelihood_value_
        )

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
