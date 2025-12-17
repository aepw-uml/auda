from typing import Tuple, override

import numpy as np
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
    output_specs=[Spec.NORMALIZED_DATASET],
)
class ZNorm(Step):
    @override
    def run(self, standardize_y: bool, on: str) -> IOValueMap:
        from sklearn.preprocessing import StandardScaler

        dataset: Tuple[np.ndarray, np.ndarray] | None = None

        match on.upper():
            case Spec.DATASET.name:
                dataset = self.get_input(Spec.DATASET.name)
            case Spec.TRAINIING_SET.name:
                dataset = self.get_input(Spec.TRAINIING_SET.name)
            case Spec.VALIDATION_SET.name:
                dataset = self.get_input(Spec.VALIDATION_SET.name)
            case Spec.TEST_SET.name:
                dataset = self.get_input(Spec.TEST_SET.name)

        if dataset is None:
            raise ValueError(f"No dataset found for '{on}'")

        X, y = dataset
        X = StandardScaler().fit_transform(X)
        if standardize_y:
            y = StandardScaler().fit_transform(y.reshape(-1, 1)).flatten()

        # TODO: Should output the mean and standard deviation used for
        return {
            Spec.NORMALIZED_DATASET.name: (X, y),
        }
