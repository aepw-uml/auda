from dataclasses import dataclass
from typing import override

import numpy as np
from auda.step.dataset import DatasetBasedStep
from auda.step.spec import Dataset, Spec
from auda.utils.pipeline import IOValueMap, step


@dataclass(frozen=True)
class DriftModel:
    x0: float
    y0: float
    slope: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        t = X[:, 0]
        return (self.y0 + self.slope * (t - self.x0)).astype(float)


@step(
    id='MD-DRIFT',
    description='Drift baseline: line through first and last time points.',
    input_specs=[Spec.ON.optional(Spec.DATASET.name)],
    output_specs=[Spec.MODEL, Spec.INTERCEPT, Spec.COEFFICIENTS],
)
class DriftBaseline(DatasetBasedStep):
    @override
    def run(self, on: str | Dataset) -> IOValueMap:
        X, y = self.get_dataset_from_on(on)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        if X.ndim != 2 or X.shape[1] != 1:
            raise ValueError('MD-DRIFT expects a single feature.')

        order = np.argsort(X[:, 0])
        t = X[order, 0]
        yy = y[order]

        x0, y0 = float(t[0]), float(yy[0])
        x1, y1 = float(t[-1]), float(yy[-1])

        denom = x1 - x0
        slope = (y1 - y0) / denom if abs(denom) > 1e-12 else 0.0

        model = DriftModel(x0=x0, y0=y0, slope=float(slope))

        # For consistency with other linear-style models:
        # y = intercept + coef * year, where intercept = y0 - slope*x0
        intercept = float(y0 - slope * x0)
        coef = float(slope)

        return {
            Spec.MODEL.name: model,
            Spec.INTERCEPT.name: intercept,
            Spec.COEFFICIENTS.name: np.array([coef], dtype=float),
        }
