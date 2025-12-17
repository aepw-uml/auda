from typing import cast, override

import numpy as np
from auda.step.spec import Spec, SpecName
from auda.utils.pipeline import IOValueMap, Step, step


@step(
    id='ST-BASIC',
    description='Computes basic descriptive statistics of the input samples.',
    input_specs=[Spec.DATASET],
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
class BasicStats(Step):
    @override
    def run(self, dataset: np.ndarray) -> IOValueMap:
        X, y = dataset
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
            SpecName.NUM_SAMPLES: num_samples,
            SpecName.NUM_FEATURES: num_features,
            SpecName.X_MEAN: x_mean,
            SpecName.X_STD: x_std,
            SpecName.X_MINIMUM: x_minimum,
            SpecName.X_MAXIMUM: x_maximum,
            SpecName.Y_MEAN: y_mean,
            SpecName.Y_STD: y_std,
            SpecName.Y_MINIMUM: y_minimum,
            SpecName.Y_MAXIMUM: y_maximum,
        }
