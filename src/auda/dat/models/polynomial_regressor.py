from typing import List, Tuple, override

import numpy as np

from auda.dat.datasets import LabeledSamples
from auda.dat.transformers import standardize_x
from auda.utils.pipeline import IOSpec, Task, task

from .__common import (
    MODEL_KIND,
    ModelISName,
    ModelOSName,
    ModelType,
    powers_2d,
    split_labeled_samples,
)


@task(
    id='MD-PR',
    kind=MODEL_KIND,
    description='Trains a Polynomial Regression model to fit nonlinear trends.',
    input_specs={
        ModelISName.SAMPLES: IOSpec(dtype=LabeledSamples),
        ModelOSName.INLIER_SAMPLES: IOSpec(dtype=LabeledSamples, required=False),
        ModelISName.DEGREE: IOSpec(dtype=int, required=False, default=4),
        ModelISName.X_MEAN: IOSpec(dtype=List[float]),
        ModelISName.X_STANDARD_DEVIATION: IOSpec(dtype=List[float]),
    },
    output_specs={
        ModelOSName.MODEL_TYPE: IOSpec(dtype=str),
        ModelOSName.INTERCEPT: IOSpec(dtype=float),
        ModelOSName.COEFFICIENTS: IOSpec(dtype=List[float]),
    },
)
class PolynomialRegressorModel(Task):
    @override
    def run(self) -> None:
        """
        Performs polynomial regression using the Ordinary Least Squares (OLS) method.
        """
        degree: int = int(self.get_input(ModelISName.DEGREE))

        samples = self.get_input(ModelISName.SAMPLES)
        inlier_samples = self.get_input(ModelISName.INLIER_SAMPLES)
        if inlier_samples is not None:
            samples = inlier_samples

        x, y = split_labeled_samples(samples)
        x = x.ravel()
        x_stdandardized = standardize_x(self, x)

        # ---- Create polynomial features
        x_poly = np.vander(x_stdandardized, N=degree + 1, increasing=True)

        # ---- Perform OLS using pseudo-inverse for numerical stability
        theta_best = np.linalg.pinv(x_poly.T @ x_poly) @ x_poly.T @ y

        # ---- Extract parameters
        intercept = float(theta_best[0])
        coefficients = theta_best[1:].tolist()

        # ---- Populate outputs
        self.set_output(ModelOSName.MODEL_TYPE, ModelType.POLYNOMIAL_REGRESSION)
        self.set_output(ModelOSName.INTERCEPT, intercept)
        self.set_output(ModelOSName.COEFFICIENTS, coefficients)


@task(
    id='MD-PR-2D',
    kind=MODEL_KIND,
    description='Trains a 2D Polynomial Regression model with cross terms.',
    input_specs={
        ModelISName.SAMPLES: IOSpec(dtype=LabeledSamples),
        ModelOSName.INLIER_SAMPLES: IOSpec(dtype=LabeledSamples, required=False),
        ModelISName.DEGREE: IOSpec(dtype=int, required=False, default=3),
        ModelISName.X_MEAN: IOSpec(dtype=List[float]),
        ModelISName.X_STANDARD_DEVIATION: IOSpec(dtype=List[float]),
    },
    output_specs={
        ModelOSName.MODEL_TYPE: IOSpec(dtype=str),
        ModelOSName.INTERCEPT: IOSpec(dtype=float),
        ModelOSName.COEFFICIENTS: IOSpec(dtype=List[float]),
        ModelOSName.COEFFICIENTS_EXPONENTS: IOSpec(dtype=List[Tuple[int]]),
    },
)
class PolynomialRegressor2D(Task):
    @override
    def run(self) -> None:
        """
        Fits z = Σ_{i+j<=D} θ_{i,j} * x^i * y^j on standardized features,
        using OLS (pseudo-inverse). Intercept is θ_{0,0}.
        """
        # ---- Params
        degree: int = int(self.get_input(ModelISName.DEGREE))

        # ---- Data (support inliers if provided)
        samples = self.get_input(ModelISName.SAMPLES)
        inlier_samples = self.get_input(ModelISName.INLIER_SAMPLES)
        if inlier_samples is not None:
            samples = inlier_samples

        X_raw, y = split_labeled_samples(samples)  # X_raw: (n, m)
        if X_raw.ndim != 2 or X_raw.shape[1] != 2:
            raise ValueError(
                f'MD-PR-2D expects exactly two features (x, y); got shape '
                f'{X_raw.shape}.'
            )

        # ---- Standardize features (broadcast via provided stats)
        X_std = standardize_x(self, X_raw)  # shape (n, 2)
        x = X_std[:, 0]
        y_std = X_std[:, 1]

        # ---- Design matrix Φ
        # Column 0 is the intercept term (i = j = 0), followed by all monomials in the
        # ordering documented in _powers_2d.
        exps = powers_2d(degree)
        n = X_std.shape[0]
        Phi = np.empty((n, len(exps)), dtype=float)

        for k, (i, j) in enumerate(exps):
            if i == 0 and j == 0:
                Phi[:, k] = 1.0  # intercept
            else:
                Phi[:, k] = (x**i) * (y_std**j)

        # ---- Solve OLS via pseudo-inverse
        theta = np.linalg.pinv(Phi.T @ Phi) @ (Phi.T @ y)

        # ---- Get coefficients (intercept excluded), intercept, and exponents
        coeffs = theta[1:].astype(float).tolist()
        intercept = float(theta[0])
        coeff_exps = [list(t) for t in exps[1:]]

        # ---- Populate outputs
        self.set_output(ModelOSName.MODEL_TYPE, ModelType.POLYNOMIAL_REGRESSION)
        self.set_output(ModelOSName.INTERCEPT, intercept)
        self.set_output(ModelOSName.COEFFICIENTS, coeffs)
        self.set_output(ModelOSName.COEFFICIENTS_EXPONENTS, coeff_exps)
