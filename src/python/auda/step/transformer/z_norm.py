from typing import override

from auda.step import get_dataset_from_step
from auda.step.spec import Spec
from auda.utils.pipeline import IOValueMap, Step, step


@step(
    id='TF-Z-NORM',
    description='Applies Z-Normalization to the dataset features and/or to '
    'the targets.',
    input_specs=[
        Spec.ON,
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
        Spec.Y_MEAN,
        Spec.Y_STD,
    ],
)
class ZNorm(Step):
    @override
    def run(self, standardize_y: bool, on: str) -> IOValueMap:
        from sklearn.preprocessing import StandardScaler

        X, y = get_dataset_from_step(self, on)

        x_scaler = StandardScaler()
        X = x_scaler.fit_transform(X)

        if standardize_y:
            y_scaler = StandardScaler()
            y = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        else:
            y_scaler = None

        return {
            Spec.NORMALIZED_DATASET.name: (X, y),
            Spec.X_MEAN.name: x_scaler.mean_,
            Spec.X_STD.name: x_scaler.scale_,
            Spec.Y_MEAN.name: y_scaler.mean_ if y_scaler else None,
            Spec.Y_STD.name: y_scaler.scale_ if y_scaler else None,
        }
