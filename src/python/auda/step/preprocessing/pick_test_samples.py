from typing import List, override

import numpy as np
from auda.step.dataset import DatasetBasedStep
from auda.step.spec import Dataset, Spec
from auda.utils.pipeline import IOValueMap, step


@step(
    id='PP-PTS',
    description='Picks specified test samples from a dataset.',
    input_specs=[
        Spec.ON.optional(Spec.DATASET.name),
        Spec.TEST_SAMPLE_INDEXES.optional([]),
    ],
    output_specs=[
        Spec.TRAINIING_SET,
        Spec.TEST_SET,
    ],
)
class PickTestSamples(DatasetBasedStep):
    @override
    def run(
        self,
        on: str | Dataset,
        test_sample_indexes: List[int] = [],
    ) -> IOValueMap:
        dataset = self.get_dataset_from_on(on)
        num_samples = self.get_num_samples(dataset)

        test_indices = np.asarray(test_sample_indexes, dtype=int)
        if test_indices.size == 0:
            if self.is_dataset_labeled(dataset):
                X, y = dataset
                empty_X = X[:0]
                empty_y = y[:0]
                return {
                    Spec.TRAINIING_SET.name: (X, y),
                    Spec.TEST_SET.name: (empty_X, empty_y),
                }
            X = dataset
            empty_X = X[:0]
            return {
                Spec.TRAINIING_SET.name: X,
                Spec.TEST_SET.name: empty_X,
            }

        if np.any(test_indices < 0) or np.any(test_indices >= num_samples):
            raise ValueError(
                'test_sample_indexes must be within dataset bounds.'
            )

        if len(np.unique(test_indices)) != len(test_indices):
            raise ValueError('test_sample_indexes must be unique.')

        mask = np.ones(num_samples, dtype=bool)
        mask[test_indices] = False
        train_indices = np.nonzero(mask)[0]

        if self.is_dataset_labeled(dataset):
            X, y = dataset
            training_set = (X[train_indices], y[train_indices])
            test_set = (X[test_indices], y[test_indices])
        else:
            X = dataset
            training_set = X[train_indices]
            test_set = X[test_indices]

        return {
            Spec.TRAINIING_SET.name: training_set,
            Spec.TEST_SET.name: test_set,
        }
