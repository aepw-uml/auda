from typing import cast, override

import numpy as np
from auda.step.dataset import DatasetStep
from auda.step.spec import Dataset, Spec
from auda.utils.pipeline import IOValueMap, step


@step(
    id='ST-BASIC',
    description='Computes basic descriptive statistics of the input samples.',
    input_specs=[Spec.ON],
    output_specs=[
        Spec.NUM_SAMPLES,
        Spec.NUM_FEATURES,
        Spec.X_MEAN,
        Spec.X_STD,
        Spec.X_MINIMUM,
        Spec.X_MAXIMUM,
        Spec.Y_MEAN,
        Spec.Y_STD,
        Spec.Y_MINIMUM,
        Spec.Y_MAXIMUM,
    ],
)
class BasicStats(DatasetStep):
    @override
    def run(self, on: str | Dataset) -> IOValueMap:
        X, y = self.get_dataset_from_on(on)
        y = cast(np.ndarray, y)

        # Statistics
        num_samples: int = len(X)
        num_features: int = len(X[0])
        x_mean: np.ndarray = np.mean(X, axis=0)
        x_std: np.ndarray = np.std(X, axis=0)
        x_minimum: np.ndarray = np.min(X, axis=0)
        x_maximum: np.ndarray = np.max(X, axis=0)
        y_mean: np.ndarray = np.mean(y, axis=0)
        y_std: np.ndarray = np.std(y, axis=0)
        y_minimum: np.ndarray = np.min(y)
        y_maximum: np.ndarray = np.max(y)

        return {
            Spec.NUM_SAMPLES.name: num_samples,
            Spec.NUM_FEATURES.name: num_features,
            Spec.X_MEAN.name: x_mean,
            Spec.X_STD.name: x_std,
            Spec.X_MINIMUM.name: x_minimum,
            Spec.X_MAXIMUM.name: x_maximum,
            Spec.Y_MEAN.name: y_mean,
            Spec.Y_STD.name: y_std,
            Spec.Y_MINIMUM.name: y_minimum,
            Spec.Y_MAXIMUM.name: y_maximum,
        }
