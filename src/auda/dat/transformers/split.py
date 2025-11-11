from typing import override

from auda.dat.datasets import LabeledSample
from auda.utils.pipeline import IOSpec, Task, task

from .__common import TRANSFORMER_KIND, TransformerISName, TransformerOSName


@task(
    id='TF-SPLIT',
    kind=TRANSFORMER_KIND,
    description='Splits labeled samples into training, validation, and test subsets.',
    input_specs={
        TransformerISName.SAMPLES: IOSpec(dtype=LabeledSample),
        TransformerISName.TRAINING_FRACTION: IOSpec(
            dtype=float, required=False, default=0.8
        ),
        TransformerISName.VALIDATION_FRACTION: IOSpec(
            dtype=float, required=False, default=0.0
        ),
    },
    output_specs={
        TransformerOSName.SAMPLES: IOSpec(dtype=LabeledSample),
        TransformerOSName.TRAIN_SAMPLES: IOSpec(dtype=LabeledSample),
        TransformerOSName.VALIDATION_SAMPLES: IOSpec(dtype=LabeledSample),
        TransformerOSName.TEST_SAMPLES: IOSpec(dtype=LabeledSample),
        TransformerOSName.NUM_TRAIN_SAMPLES: IOSpec(dtype=int),
        TransformerOSName.NUM_VALIDATION_SAMPLES: IOSpec(dtype=int),
        TransformerOSName.NUM_TEST_SAMPLES: IOSpec(dtype=int),
    },
)
class TrimmingTransformer(Task):
    @override
    def run(self) -> None:
        # ---- Parameters
        training_fraction = self.get_input(TransformerISName.TRAINING_FRACTION)
        validation_fraction = self.get_input(TransformerISName.VALIDATION_FRACTION)

        if training_fraction + validation_fraction > 1.0:
            raise ValueError(
                'The sum of training_fraction and validation_fraction must be '
                'less than or equal to 1.0'
            )

        samples = self.get_input(TransformerISName.SAMPLES)
        n_samples = len(samples)

        n_training_samples = int(n_samples * training_fraction)
        n_validation_samples = int(n_samples * validation_fraction)
        training_samples = samples[:n_training_samples]
        validation_samples = samples[
            n_training_samples : n_training_samples + n_validation_samples
        ]
        testing_samples = samples[n_training_samples - n_validation_samples :]

        # ---- Populate outputs
        self.set_output(TransformerOSName.SAMPLES, training_samples)
        self.set_output(TransformerOSName.TRAIN_SAMPLES, training_samples)
        self.set_output(TransformerOSName.VALIDATION_SAMPLES, validation_samples)
        self.set_output(TransformerOSName.TEST_SAMPLES, testing_samples)
        self.set_output(TransformerOSName.NUM_TRAIN_SAMPLES, n_training_samples)
        self.set_output(TransformerOSName.NUM_VALIDATION_SAMPLES, n_validation_samples)
        self.set_output(TransformerOSName.NUM_TEST_SAMPLES, len(testing_samples))
