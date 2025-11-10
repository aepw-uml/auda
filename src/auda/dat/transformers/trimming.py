from typing import override

from auda.dat.datasets import LabeledSample
from auda.utils.pipeline import IOSpec, Task, task

from .__common import TRANSFORMER_KIND, TransformerISName, TransformerOSName


@task(
    id='TF-TRIMMING',
    kind=TRANSFORMER_KIND,
    description='Removes samples with the smallest and largest univariate feature '
    'values.',
    input_specs={
        TransformerISName.SAMPLES: IOSpec(dtype=LabeledSample),
        TransformerISName.LOWER_TRIMMING_PERCENTAGE: IOSpec(
            dtype=float, required=False, default=0.1
        ),
        TransformerISName.UPPER_TRIMMING_PERCENTAGE: IOSpec(
            dtype=float, required=False, default=0.1
        ),
    },
    output_specs={
        TransformerOSName.SAMPLES: IOSpec(dtype=LabeledSample),
        TransformerOSName.TRIMMED_SAMPLES: IOSpec(dtype=LabeledSample),
    },
)
class TrimmingTransformer(Task):
    @override
    def run(self) -> None:
        samples = self.get_input(TransformerISName.SAMPLES)
        if len(samples[0][0]) != 1:
            raise ValueError('TrimmingTransformer only supports univariate samples.')

        lower_percentage = float(
            self.get_input(TransformerISName.LOWER_TRIMMING_PERCENTAGE)
        )
        upper_percentage = float(
            self.get_input(TransformerISName.UPPER_TRIMMING_PERCENTAGE)
        )

        # ---- Calculate thresholds
        x_vals = [sample[0][0] for sample in samples]
        x_minimum = min(x_vals)
        x_maximum = max(x_vals)
        x_diff = x_maximum - x_minimum
        x_lower_threshold = x_minimum + lower_percentage * x_diff
        x_upper_threshold = x_maximum - upper_percentage * x_diff

        trimmed_samples = [
            sample
            for sample in samples
            if x_lower_threshold <= sample[0][0] <= x_upper_threshold
        ]

        # ---- Populate outputs
        self.set_output(TransformerOSName.SAMPLES, trimmed_samples)
        self.set_output(TransformerOSName.TRIMMED_SAMPLES, trimmed_samples)
