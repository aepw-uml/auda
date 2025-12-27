from dataclasses import dataclass
from typing import override

import numpy as np
from auda.step.dataset import DatasetBasedStep
from auda.step.spec import Dataset, Spec
from auda.utils.pipeline import IOValueMap, step


@dataclass(frozen=True)
class NaiveLastValueModel:
    last_y: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        n = int(X.shape[0])
        return np.full((n,), self.last_y, dtype=float)


@step(
    id='MD-NAIVE',
    description='Naïve (persistence) baseline: predicts the last observed y.',
    input_specs=[Spec.ON.optional(Spec.DATASET.name)],
    output_specs=[Spec.MODEL, Spec.INTERCEPT],
)
class NaiveLastValue(DatasetBasedStep):
    @override
    def run(self, on: str | Dataset) -> IOValueMap:
        X, y = self.get_dataset_from_on(on)

        if X.ndim != 2 or X.shape[1] != 1:
            raise ValueError('MD-NAIVE expects a single feature.')

        # Sort by time, take last observed value
        order = np.argsort(X[:, 0])
        last_y = float(y[order][-1])

        model = NaiveLastValueModel(last_y=last_y)

        return {
            Spec.MODEL.name: model,
            Spec.INTERCEPT.name: last_y,
        }
