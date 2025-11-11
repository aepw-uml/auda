from typing import List, override

from auda.dat.datasets import LabeledSamples
from auda.dat.models import split_labeled_samples
from auda.dat.transformers import standardize_x
from auda.utils.pipeline import IOSpec, Task, task

from .__common import EvaluatorISName, EvaluatorOSName


@task(
    id='MD-PR-2D-EVAL',
    kind='model',
    description='Evaluates a trained 2D polynomial regression model on test samples.',
    input_specs={
        EvaluatorISName.TEST_SAMPLES: IOSpec(dtype=LabeledSamples),
        EvaluatorISName.INTERCEPT: IOSpec(dtype=float),
        EvaluatorISName.COEFFICIENTS: IOSpec(dtype=List[float]),
        EvaluatorISName.COEFFICIENTS_EXPONENTS: IOSpec(dtype=List[List[int]]),
        EvaluatorISName.DEGREE: IOSpec(dtype=int),
        EvaluatorISName.X_MEAN: IOSpec(dtype=List[float]),
        EvaluatorISName.X_STANDARD_DEVIATION: IOSpec(dtype=List[float]),
    },
    output_specs={
        EvaluatorOSName.Y_TRUE: IOSpec(dtype=List[float]),
        EvaluatorOSName.Y_PRED: IOSpec(dtype=List[float]),
        EvaluatorOSName.RESIDUALS: IOSpec(dtype=List[float]),
        EvaluatorOSName.R2: IOSpec(dtype=float),
        EvaluatorOSName.ROOT_MEAN_SQUARED_ERROR: IOSpec(dtype=float),
        EvaluatorOSName.MEAN_ABSOLUTE_ERROR: IOSpec(dtype=float),
        EvaluatorOSName.MEAN_ABSOLUTE_PERCENTAGE_ERROR: IOSpec(dtype=float),
    },
)
class PolynomialRegressor2DEval(Task):
    @override
    def run(self) -> None:
        import numpy as np

        intercept: float = float(self.get_input(EvaluatorISName.INTERCEPT))
        coeffs: List[float] = self.get_input(EvaluatorISName.COEFFICIENTS)
        exps: List[List[int]] = self.get_input(EvaluatorISName.COEFFICIENTS_EXPONENTS)

        # ---- Ensure (0,0) intercept term is first in exponents
        if not exps or tuple(exps[0]) != (0, 0):
            exps = [[0, 0]] + exps  # model coeffs are aligned to exps[1:]

        # ---- Data preparation
        test_samples = self.get_input(EvaluatorISName.TEST_SAMPLES)
        X_raw, y = split_labeled_samples(test_samples)
        X_std = standardize_x(self, X_raw)
        x = X_std[:, 0]
        y2 = X_std[:, 1]

        n = X_std.shape[0]
        m = len(exps)

        # Early out if no data
        if n == 0:
            self.set_output(EvaluatorOSName.Y_TRUE, [])
            self.set_output(EvaluatorOSName.Y_PRED, [])
            self.set_output(EvaluatorOSName.RESIDUALS, [])
            self.set_output(EvaluatorOSName.R2, float('nan'))
            self.set_output(EvaluatorOSName.ROOT_MEAN_SQUARED_ERROR, float('nan'))
            self.set_output(EvaluatorOSName.MEAN_ABSOLUTE_ERROR, float('nan'))
            self.set_output(
                EvaluatorOSName.MEAN_ABSOLUTE_PERCENTAGE_ERROR, float('nan')
            )
            return

        # ---- Design matrix Φ (column 0 = intercept)
        Phi = np.empty((n, m), dtype=float)
        for k, (i, j) in enumerate(exps):
            if i == 0 and j == 0:
                Phi[:, k] = 1.0
            else:
                Phi[:, k] = (x**i) * (y2**j)

        # ---- Parameter vector θ aligned with exps
        theta = np.empty(m, dtype=float)
        theta[0] = intercept
        theta[1:] = np.asarray(coeffs, dtype=float)

        # ---- Predictions & residuals
        y_pred = Phi @ theta
        resid = y_pred - y

        # ---- Metrics
        ss_res = float(np.sum((y - y_pred) ** 2))
        y_mean = float(np.mean(y))
        ss_tot = float(np.sum((y - y_mean) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

        rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
        mae = float(np.mean(np.abs(y - y_pred)))

        # MAPE with zero-safe denominator
        denom = np.where(np.abs(y) < 1e-12, np.nan, np.abs(y))
        mape = float(np.nanmean(np.abs((y - y_pred) / denom) * 100.0))

        # ---- Outputs
        self.set_output(EvaluatorOSName.Y_TRUE, y.astype(float).tolist())
        self.set_output(EvaluatorOSName.Y_PRED, y_pred.astype(float).tolist())
        self.set_output(EvaluatorOSName.RESIDUALS, resid.astype(float).tolist())
        self.set_output(EvaluatorOSName.R2, r2)
        self.set_output(EvaluatorOSName.ROOT_MEAN_SQUARED_ERROR, rmse)
        self.set_output(EvaluatorOSName.MEAN_ABSOLUTE_ERROR, mae)
        self.set_output(EvaluatorOSName.MEAN_ABSOLUTE_PERCENTAGE_ERROR, mape)
