from typing import List, override

import numpy as np

from auda.dat.datasets import LabeledSamples
from auda.dat.transformers import standardize_x
from auda.utils.pipeline import IOSpec, Task, task

from .__common import (
    MODEL_KIND,
    ModelISName,
    ModelOSName,
    ModelType,
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
