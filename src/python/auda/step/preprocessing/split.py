from typing import cast, override

import numpy as np
from auda.step.dataset import DatasetBasedStep
from auda.step.spec import Dataset, LabeledDataset, Spec, UnlabeledDataset
from auda.utils.pipeline import IOValueMap, step
from diskcache.recipes import math


@step(
    id='PP-SPLIT',
    description='Splits a dataset into training and testing sets based on the '
    'specified ',
    input_specs=[
        Spec.ON.optional(Spec.DATASET.name),
        Spec.TRAINING_PORTION.optional(0.8),
        Spec.SHOULD_SHUFFLE.optional(False),
        Spec.SEED.optional(42),
    ],
    output_specs=[
        Spec.TRAINIING_SET,
        Spec.TEST_SET,
        Spec.NUM_TRAINING_SAMPLES,
        Spec.NUM_TEST_SAMPLES,
    ],
)
class Split(DatasetBasedStep):
    @override
    def run(
        self,
        on: str | Dataset,
        training_portion: float,
        should_shuffle: bool,
        seed: int,
    ) -> IOValueMap:
        dataset = self.get_dataset_from_on(on)
        num_samples = self.get_num_samples(dataset)
        is_labeled = self.is_dataset_labeled(dataset)

        indices = np.arange(num_samples)

        if should_shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)

        split_idx = math.ceil(num_samples * training_portion)
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        if is_labeled:
            X, y = cast(LabeledDataset, dataset)
            training_set = (X[train_indices], y[train_indices])
            test_set = (X[test_indices], y[test_indices])
        else:
            X = cast(UnlabeledDataset, dataset)
            training_set = X[train_indices]
            test_set = X[test_indices]

        return {
            Spec.TRAINIING_SET.name: training_set,
            Spec.TEST_SET.name: test_set,
            Spec.NUM_TRAINING_SAMPLES.name: len(train_indices),
            Spec.NUM_TEST_SAMPLES.name: len(test_indices),
        }
