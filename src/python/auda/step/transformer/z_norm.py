from typing import Tuple, override

import numpy as np
from auda.step import get_dataset_from_step
from auda.step.spec import Dataset, Spec
from auda.utils.pipeline import IOValueMap, Step, step


@step(
    id='TF-Z-NORM',
    description='Applies Z-Normalization to the dataset features and/or to '
    'the targets.',
    input_specs=[
        Spec.ON.optional(Spec.DATASET.name),
        Spec.STANDARDIZE_Y.optional(True),
        Spec.DATASET.optional(),
        Spec.TRAINIING_SET.optional(),
        Spec.VALIDATION_SET.optional(),
        Spec.TEST_SET.optional(),
    ],
    output_specs=[
        Spec.NORMALIZED_DATASET,
        Spec.X_MEAN,
        Spec.X_STD,
        Spec.Y_MEAN.optional(),
        Spec.Y_STD.optional(),
    ],
)
class ZNorm(Step):
    @override
    def run(self, on: str | Dataset, standardize_y: bool) -> IOValueMap:
        X, y = get_dataset_from_step(self, on)
        X, X_mean, X_std = self._z_normalize(X)

        if standardize_y:
            y_scaled, y_mean, y_std = self._z_normalize(y)
        else:
            y_scaled = None

        return {
            Spec.NORMALIZED_DATASET.name: (X, y_scaled if standardize_y else y),
            Spec.X_MEAN.name: X_mean,
            Spec.X_STD.name: X_std,
            Spec.Y_MEAN.name: y_mean if standardize_y else None,
            Spec.Y_STD.name: y_std if standardize_y else None,
        }

    def _z_normalize(
        self, data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std_replaced = np.where(std == 0, 1, std)
        normalized_data = (data - mean) / std_replaced

        return normalized_data, mean, std
