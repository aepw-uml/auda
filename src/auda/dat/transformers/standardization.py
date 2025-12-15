from typing import Union, override

from numpy import ndarray

from auda.dat import run_pipeline
from auda.dat.datasets import LabeledSamples, UnlabeledSamples
from auda.utils.pipeline import IOSpec, Task, task

from .__common import TRANSFORMER_KIND, TransformerISName, TransformerOSName


@task(
    id='TF-Z-NORM',
    kind=TRANSFORMER_KIND,
    description='Standardizes features (and optionally labels) to zero mean and unit '
    'variance.',
    input_specs={
        TransformerISName.SAMPLES: IOSpec(dtype=LabeledSamples),
        TransformerISName.TO_STANDARDIZE_Y: IOSpec(
            dtype=int, required=False, default=0
        ),
    },
    output_specs={
        TransformerOSName.X_STANDARDIZED: IOSpec(dtype=ndarray),
        TransformerOSName.Y_STANDARDIZED: IOSpec(dtype=ndarray),
    },
)
class StandardizationTransformer(Task):
    @override
    def run(self) -> None:
        """
        Standardizes the input samples to have zero mean and unit variance.
        """
        import numpy as np
        from sklearn.preprocessing import StandardScaler

        samples: Union[LabeledSamples, UnlabeledSamples] = self.get_input(
            TransformerISName.SAMPLES
        )

        # Scale x
        is_labeled_samples = isinstance(samples[0], tuple)
        x = (
            np.array([x for x, _ in samples])
            if is_labeled_samples
            else np.array(samples)
        )
        x_standardized = StandardScaler().fit_transform(x)
        self.set_output(TransformerOSName.X_STANDARDIZED, x_standardized)

        # Scale y
        to_standardize_y = self.get_input(TransformerISName.TO_STANDARDIZE_Y)
        if is_labeled_samples and bool(to_standardize_y):
            y = np.array([y for _, y in samples]).reshape(-1, 1)
            y_standardized = StandardScaler().fit_transform(y).flatten()
            self.set_output(TransformerOSName.Y_STANDARDIZED, y_standardized)

        # ---- Populate outputs
        self.set_output(TransformerOSName.X_STANDARDIZED, x_standardized)


@task(
    id='TF-Z-NORM-TEST',
    kind=TRANSFORMER_KIND,
    description='Standardizes features (and optionally labels) to zero mean and unit '
    'variance (on test samples).',
    input_specs={
        TransformerISName.TEST_SAMPLES: IOSpec(dtype=LabeledSamples),
        TransformerISName.X_MEAN: IOSpec(dtype=int),
        TransformerISName.X_STANDARD_DEVIATION: IOSpec(dtype=int),
    },
    output_specs={
        TransformerOSName.ORIGINAL_TEST_SAMPLES: IOSpec(dtype=LabeledSamples),
        TransformerOSName.TEST_SAMPLES: IOSpec(dtype=LabeledSamples),
    },
)
class TestStandardizationTransformer(Task):
    @override
    def run(self) -> None:
        """
        Standardizes the test samples to have zero mean and unit variance.
        """
        samples = self.get_input(TransformerISName.TEST_SAMPLES)
        outputs, _ = run_pipeline(
            ['TF-Z-NORM'],
            {
                TransformerISName.SAMPLES: samples,
            },
        )

        # ---- Populate outputs
        self.set_output(TransformerOSName.ORIGINAL_TEST_SAMPLES, samples)
        self.set_output(
            TransformerOSName.TEST_SAMPLES,
            outputs[TransformerOSName.X_STANDARDIZED],
        )
